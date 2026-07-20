"""
explain.py
----------
YBU ilac dozu izleme sistemi icin acikanabilirlik (explainability) katmani.

Sorumluluklari:
    - Son tahmin icin KernelSHAP ozellik katkilari
    - Son tahmin icin LIME ozellik katkilari
    - Anomali noktasi ile normal nokta icin SHAP katkilarinin karsilastirilmasi
    - Claude API (Anthropic) ile SHAP bulgularina dayali Turkce klinik yorum uretimi

Not: forecast.py'deki ARIMA/LSTM hibrit modeli dogrudan SHAP/LIME ile
aciklanamayacak kadar heterojen (iki farkli model + agirlikli birlesim)
oldugundan, bu modul; gecikme (lag) ve kayan istatistik ozelliklerinden
bir sonraki dozu tahmin eden, sklearn tabanli yorumlanabilir bir "vekil"
(surrogate) regresyon modeli uzerinden calisir. Bu, klinik pratikte de
yaygin bir yaklasimdir: karmasik/hibrit modelin davranisini, ayni girdi
uzayinda egitilen basit bir modelle yaklasik olarak acikliga kavustururuz.

Bagimsiz calistirma:
    python explain.py
"""

from __future__ import annotations

import json
import os

import anthropic
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

FEATURE_LAGS = (1, 2, 3, 6, 12)
CLINICAL_COMMENTARY_MODEL = "claude-sonnet-4-6"
CLINICAL_COMMENTARY_MAX_TOKENS = 500
AI_ANALYSIS_MODEL = "claude-sonnet-4-6"
AI_ANALYSIS_MAX_TOKENS = 800

# Kural tabanli yorum icin ozellik adi -> okunabilir aciklama eslemesi
_FEATURE_DESCRIPTIONS = {
    "lag_1": "1 saat onceki doz degeri",
    "lag_2": "2 saat onceki doz degeri",
    "lag_3": "3 saat onceki doz degeri",
    "lag_6": "6 saat onceki doz degeri",
    "lag_12": "12 saat onceki doz degeri",
    "rolling_mean": "son 6 saatlik kayan ortalama",
    "rolling_std": "son 6 saatlik kayan degiskenlik (std)",
}

# Kural tabanli yorum icin ilaca ve doz yonune (artis/azalis) gore klinik oneri
_DRUG_GUIDANCE = {
    "norepinephrine": {
        "above": "Vazopressor dozundaki artis nedeniyle kan basinci hedefi ve asiri "
                 "vazokonstriksiyon bulgulari (ekstremite perfuzyonu, laktat) gozden gecirilmelidir.",
        "below": "Doz azalisi hipotansiyon riski tasiyabilir; MAP takibi siklastirilmali "
                 "ve titrasyon plani yeniden degerlendirilmelidir.",
    },
    "propofol": {
        "above": "Sedasyon derinligindeki artis nedeniyle solunum durumu ve hemodinamik "
                 "stabilite yakindan izlenmelidir.",
        "below": "Sedasyon yetersizligi ajitasyon veya planlanmamis ekstubasyon riski "
                 "dogurabilir; sedasyon skoru (orn. RASS) yeniden degerlendirilmelidir.",
    },
    "insulin": {
        "above": "Insulin dozundaki artis hipoglisemi riski tasir; kan glukozu takibi siklastirilmalidir.",
        "below": "Doz azalisi hiperglisemiye yol acabilir; glukoz takibi ve insulin "
                 "protokolu gozden gecirilmelidir.",
    },
    "heparin": {
        "above": "Antikoagulan dozundaki artis kanama riskini yukseltebilir; aPTT/anti-Xa "
                 "takibi ve kanama bulgulari degerlendirilmelidir.",
        "below": "Doz azalisi tromboz riskini artirabilir; antikoagulasyon hedefi ve "
                 "aPTT sonuclari gozden gecirilmelidir.",
    },
}


# --------------------------------------------------------------------------
# Ozellik muhendisligi + vekil model
# --------------------------------------------------------------------------

def build_feature_frame(series: pd.Series, lags: tuple = FEATURE_LAGS,
                         roll_window: int = 6) -> pd.DataFrame:
    """Gecikme ve kayan istatistik ozelliklerinden bir ozellik matrisi kurar.

    Ozellikler:
        lag_{k}: t-k anindaki deger (k in lags)
        rolling_mean: son roll_window saatin ortalamasi (t-1'e kadar)
        rolling_std: son roll_window saatin std'si (t-1'e kadar)

    Hedef (target) sutunu: t anindaki deger (bir sonraki doz).

    Args:
        series: Kaynak zaman serisi.
        lags: Kullanilacak gecikme adimlari.
        roll_window: Kayan istatistik pencere uzunlugu.

    Returns:
        "target" ve ozellik sutunlarini iceren, NaN satirlari atilmis DataFrame.
    """
    df = pd.DataFrame({"target": series})
    for k in lags:
        df[f"lag_{k}"] = series.shift(k)
    df["rolling_mean"] = series.shift(1).rolling(roll_window).mean()
    df["rolling_std"] = series.shift(1).rolling(roll_window).std()
    return df.dropna()


def train_surrogate_model(
    feature_df: pd.DataFrame, seed: int = 42, test_frac: float = 0.2
) -> tuple[RandomForestRegressor, list[str], dict]:
    """Ozellik matrisi uzerinde vekil bir RandomForestRegressor egitir.

    Onemli duzeltme: Onceki surumde model TUM veriyle egitilip, UI'de
    gosterilen "tahmin" de AYNI veri uzerinde hesaplaniyordu (in-sample) -
    bu, gercekte modelin gormedigi veriyi tahmin etme basarisini degil,
    ezberleme basarisini yansitir ve yaniltici derecede iyimserdir. Bu
    surumde zaman serisi oldugu icin KRONOLOJIK (karistirmasiz) bir
    train/test ayrimi yapilir: son `test_frac` kadar gozlem test icin
    ayrilir, model YALNIZCA train kismiyla egitilir; dondurulen test_metrics
    gercek bir out-of-sample (modelin gormedigi veri) performansini yansitir.

    SHAP/LIME acikanabilirligi icin bu ayrim bir kisitlama getirmez - egitilen
    model, train veya test bolgesinden herhangi bir noktayi aciklamak icin
    kullanilabilir (SHAP/LIME model-icgozlemidir, aciklanan noktanin egitimde
    kullanilmis olmasini gerektirmez); onemli olan raporlanan DOGRULUK
    metriginin tutulmamis (held-out) veriden gelmesidir.

    Args:
        feature_df: build_feature_frame ciktisi ("target" + ozellik sutunlari).
        seed: Rastgelelik tohumu.
        test_frac: Test icin ayrilacak (serinin sonundaki) gozlem orani.

    Returns:
        (egitilmis model, ozellik sutun adlari listesi, test_metrics)
        test_metrics: {"mae": float, "rmse": float, "n_train": int, "n_test": int}
        ya da veri cok azsa (test seti < 3 gozlem) {"warning": str} - bu
        durumda model tum veriyle egitilir ve out-of-sample metrik
        raporlanamayacagi acikca isaretlenir (in-sample sonuc, out-of-sample
        gibi sunulmaz).
    """
    feature_cols = [c for c in feature_df.columns if c != "target"]
    n = len(feature_df)
    n_test = int(n * test_frac)

    if n_test < 3:
        X = feature_df[feature_cols].to_numpy()
        y = feature_df["target"].to_numpy()
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed)
        model.fit(X, y)
        return model, feature_cols, {
            "warning": "Yetersiz veri: out-of-sample test ayrimi yapilamadi (n_test<3), "
                       "model tum veriyle egitildi ve dogruluk metrigi raporlanmiyor."
        }

    train_df = feature_df.iloc[:-n_test]
    test_df = feature_df.iloc[-n_test:]

    X_train, y_train = train_df[feature_cols].to_numpy(), train_df["target"].to_numpy()
    X_test, y_test = test_df[feature_cols].to_numpy(), test_df["target"].to_numpy()

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_metrics = {
        "mae": float(np.mean(np.abs(y_test - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    return model, feature_cols, test_metrics


# --------------------------------------------------------------------------
# KernelSHAP
# --------------------------------------------------------------------------

def kernelshap_explain(model: RandomForestRegressor, X_background: np.ndarray,
                        x_instance: np.ndarray, feature_names: list[str],
                        n_background: int = 30, seed: int = 42) -> dict:
    """KernelSHAP ile tek bir ornek (orn. son tahmin) icin ozellik katkilarini hesaplar.

    Args:
        model: Egitilmis regresyon modeli (predict metodu olan herhangi bir model).
        X_background: Arka plan (referans) veri kumesi, (n_samples, n_features).
        x_instance: Aciklanacak tek ornek, (n_features,) veya (1, n_features).
        feature_names: Ozellik adlari.
        n_background: KernelExplainer icin kmeans ile ozetlenecek arka plan nokta sayisi.
        seed: Rastgelelik tohumu.

    Returns:
        dict: {
            "base_value": float,
            "prediction": float,
            "contributions": {feature_name: shap_value, ...},  # buyukluge gore siralanmis
        }
    """
    x_instance = np.asarray(x_instance).reshape(1, -1)
    n_clusters = min(n_background, len(X_background), 10)
    background = shap.kmeans(X_background, n_clusters)

    explainer = shap.KernelExplainer(model.predict, background, seed=seed)
    shap_values = explainer.shap_values(x_instance, silent=True)
    shap_values = np.asarray(shap_values).reshape(-1)

    contributions = dict(zip(feature_names, shap_values.tolist()))
    contributions = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    return {
        "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        "prediction": float(model.predict(x_instance)[0]),
        "contributions": contributions,
    }


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------

def lime_explain(model: RandomForestRegressor, X_train: np.ndarray, x_instance: np.ndarray,
                  feature_names: list[str], num_features: int | None = None, seed: int = 42) -> dict:
    """LIME ile tek bir ornek icin yerel (local) ozellik katkilarini hesaplar.

    Args:
        model: Egitilmis regresyon modeli.
        X_train: LIME'in yerel komsulugu ornekleyecegi egitim verisi.
        x_instance: Aciklanacak tek ornek, (n_features,).
        feature_names: Ozellik adlari.
        num_features: Aciklamada gosterilecek ozellik sayisi; None ise tumu.
        seed: Rastgelelik tohumu.

    Returns:
        dict: {
            "prediction": float,
            "contributions": {feature_name: lime_weight, ...},  # buyukluge gore siralanmis
        }
    """
    num_features = num_features or len(feature_names)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=feature_names,
        mode="regression",
        random_state=seed,
        discretize_continuous=False,
    )
    exp = explainer.explain_instance(
        np.asarray(x_instance).reshape(-1),
        model.predict,
        num_features=num_features,
    )
    contributions = dict(exp.as_list())
    contributions = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    return {
        "prediction": float(model.predict(np.asarray(x_instance).reshape(1, -1))[0]),
        "contributions": contributions,
    }


# --------------------------------------------------------------------------
# Anomali vs. normal SHAP karsilastirmasi
# --------------------------------------------------------------------------

def compare_anomaly_vs_normal(model: RandomForestRegressor, X_background: np.ndarray,
                               x_anomaly: np.ndarray, x_normal: np.ndarray,
                               feature_names: list[str], n_background: int = 30,
                               seed: int = 42) -> pd.DataFrame:
    """Anomali noktasi ile normal nokta icin SHAP katkilarini yan yana karsilastirir.

    Args:
        model: Egitilmis regresyon modeli.
        X_background: Arka plan veri kumesi.
        x_anomaly: Anomali olarak isaretlenmis ornek.
        x_normal: Normal (referans) ornek.
        feature_names: Ozellik adlari.
        n_background: KernelExplainer arka plan ozet nokta sayisi.
        seed: Rastgelelik tohumu.

    Returns:
        Sutunlari ["feature", "shap_anomaly", "shap_normal", "abs_diff"] olan,
        abs_diff'e gore azalan siralanmis pandas.DataFrame.
    """
    anomaly_res = kernelshap_explain(model, X_background, x_anomaly, feature_names,
                                      n_background=n_background, seed=seed)
    normal_res = kernelshap_explain(model, X_background, x_normal, feature_names,
                                     n_background=n_background, seed=seed)

    rows = []
    for feat in feature_names:
        shap_a = anomaly_res["contributions"].get(feat, 0.0)
        shap_n = normal_res["contributions"].get(feat, 0.0)
        rows.append({
            "feature": feat,
            "shap_anomaly": shap_a,
            "shap_normal": shap_n,
            "abs_diff": abs(shap_a - shap_n),
        })

    comparison = pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)
    comparison.attrs["prediction_anomaly"] = anomaly_res["prediction"]
    comparison.attrs["prediction_normal"] = normal_res["prediction"]
    return comparison


# --------------------------------------------------------------------------
# Kural tabanli (rule-based) klinik yorum - Claude API GEREKTIRMEZ
# --------------------------------------------------------------------------

def generate_rule_based_commentary(
    drug_name: str,
    dose: float,
    dose_unit: str,
    anomaly_time: str,
    z_score: float | None,
    top_shap_features: list[tuple[str, float]],
    vitals: dict[str, float],
) -> str:
    """Z-skoru ve SHAP degerlerine dayali, tamamen if/else mantigiyla Turkce
    klinik yorum uretir. Herhangi bir API cagrisi veya API anahtari gerektirmez.

    Mantik:
        1. |Z-skoru| buyuklugune gore siddet derecesi belirlenir
           (sinirda / hafif / belirgin / kritik).
        2. Z-skorunun isareti, dozun beklenen araligin uzerinde mi altinda mi
           oldugunu belirler (artis / azalis).
        3. En yuksek mutlak SHAP degerine sahip ozellik, katkisinin yonuyle
           (pozitif/negatif) birlikte adlandirilir.
        4. Mevcut vital bulgular (MAP<65, HR>100 veya <60, Laktat>2.0) icin
           basit esik kontrolleriyle ek uyarilar eklenir.
        5. Ilaca ve doz yonune ozgu, onceden tanimlanmis bir klinik oneri
           (_DRUG_GUIDANCE) secilir.

    Args:
        drug_name: Ilac adi (orn. "norepinephrine").
        dose: Anomali anindaki doz degeri.
        dose_unit: Doz birimi (orn. "mcg/kg/dk").
        anomaly_time: Anomalinin zaman damgasi (okunabilir string).
        z_score: Kayan Z-Score degeri; bilinmiyorsa None.
        top_shap_features: [(ozellik_adi, shap_degeri), ...], buyukluge gore
            siralanmis en fazla ilk 3 ozellik (yalnizca ilki kullanilir).
        vitals: {"MAP": deger, "HR": deger, "Laktat": deger, ...}; eksik
            olan vitaller sozlukte bulunmayabilir.

    Returns:
        Turkce klinik yorum metni.
    """
    # 1) Siddet derecesi
    if z_score is None:
        severity = "belirsiz siddette"
        z_text = "bilinmiyor"
    else:
        abs_z = abs(z_score)
        z_text = f"{z_score:.2f}"
        if abs_z >= 4.0:
            severity = "kritik duzeyde"
        elif abs_z >= 3.0:
            severity = "belirgin"
        elif abs_z >= 2.5:
            severity = "hafif"
        else:
            severity = "sinirda (esik altinda ancak izlenmesi gereken)"

    # 2) Doz yonu
    if z_score is None:
        direction_key = "above"  # varsayilan: artis yonu esas alinir
        direction_text = "beklenen araligin disinda"
    elif z_score > 0:
        direction_key = "above"
        direction_text = "beklenen araligin uzerinde"
    else:
        direction_key = "below"
        direction_text = "beklenen araligin altinda"

    # 3) En etkili SHAP ozelligi
    if top_shap_features:
        top_name, top_value = top_shap_features[0]
        feature_desc = _FEATURE_DESCRIPTIONS.get(top_name, top_name)
        shap_direction = "tahmini yukselten yonde" if top_value >= 0 else "tahmini dusuren yonde"
        shap_sentence = (
            f"SHAP analizi {feature_desc} ozelliginin ana katki sagladigini gosteriyor "
            f"({shap_direction}, katki={top_value:+.4f})."
        )
    else:
        shap_sentence = "SHAP verisi bu nokta icin mevcut degil."

    # 4) Vital bulgu esik kontrolleri
    vital_flags = []
    if "MAP" in vitals and vitals["MAP"] < 65:
        vital_flags.append(f"MAP degeri {vitals['MAP']:.0f} mmHg ile hipotansif sinirin altindadir.")
    if "HR" in vitals:
        if vitals["HR"] > 100:
            vital_flags.append(f"Kalp hizi {vitals['HR']:.0f}/dk ile takikardik araliktadir.")
        elif vitals["HR"] < 60:
            vital_flags.append(f"Kalp hizi {vitals['HR']:.0f}/dk ile bradikardik araliktadir.")
    if "Laktat" in vitals and vitals["Laktat"] > 2.0:
        vital_flags.append(f"Laktat {vitals['Laktat']:.1f} mmol/L ile yuksektir, doku perfuzyonu degerlendirilmelidir.")
    vital_text = (" " + " ".join(vital_flags)) if vital_flags else ""

    # 5) Ilaca ozgu klinik oneri
    guidance = _DRUG_GUIDANCE.get(drug_name, {}).get(
        direction_key,
        "Doz degisimi klinik olarak degerlendirilmeli, hasta yakindan izlenmelidir.",
    )

    return (
        f"Saat {anomaly_time}'te anomali tespit edildi. {drug_name} dozu ({dose:.4f} {dose_unit}) "
        f"{severity} sekilde {direction_text} (Z-skoru: {z_text}). {shap_sentence}{vital_text} "
        f"Onerilen eylem: {guidance}"
    )


# --------------------------------------------------------------------------
# Claude API ile klinik yorum
# --------------------------------------------------------------------------

def generate_clinical_commentary(
    drug_name: str,
    dose: float,
    dose_unit: str,
    anomaly_time: str,
    z_score: float | None,
    top_shap_features: list[tuple[str, float]],
    vitals: dict[str, float],
    api_key: str,
    model: str = CLINICAL_COMMENTARY_MODEL,
    max_tokens: int = CLINICAL_COMMENTARY_MAX_TOKENS,
) -> str:
    """Tespit edilen bir doz anomalisi icin Claude API araciligiyla Turkce klinik yorum uretir.

    Modele; ilac adi/dozu, anomali zamani ve Z-skoru, en onemli 3 SHAP
    ozelligi/degeri ve mevcut vital bulgular (orn. MAP, HR, laktat) baglam
    olarak verilir; model bunlari yorumlayip kisa bir klinik degerlendirme ve
    onerilen eylem icerin Turkce bir metin uretir.

    Args:
        drug_name: Ilac adi (orn. "norepinephrine").
        dose: Anomali anindaki doz degeri.
        dose_unit: Doz birimi (orn. "mcg/kg/dk").
        anomaly_time: Anomalinin zaman damgasi (okunabilir string).
        z_score: Kayan Z-Score degeri; bilinmiyorsa None.
        top_shap_features: [(ozellik_adi, shap_degeri), ...], buyukluge gore
            siralanmis en fazla ilk 3 ozellik.
        vitals: {"MAP": deger, "HR": deger, "Laktat": deger, ...}; eksik
            olan vitaller sozlukte bulunmayabilir.
        api_key: Anthropic API anahtari (orn. st.secrets["ANTHROPIC_API_KEY"]).
        model: Kullanilacak Claude modeli.
        max_tokens: Yanit icin azami token sayisi.

    Returns:
        Claude'un urettigi Turkce klinik yorum metni.

    Raises:
        RuntimeError: API cagrisi basarisiz olursa (kimlik dogrulama, hiz
            siniri, sunucu/aglantı hatasi) aciklayici bir Turkce mesajla.
    """
    shap_lines = "\n".join(
        f"  - {name}: {value:+.4f}" for name, value in top_shap_features
    ) or "  (SHAP verisi mevcut degil)"

    vital_lines = "\n".join(
        f"  - {name}: {value:.1f}" for name, value in vitals.items()
    ) or "  (vital bulgu verisi mevcut degil)"

    z_score_text = f"{z_score:.2f}" if z_score is not None else "bilinmiyor"

    prompt = f"""Sen bir Yogun Bakim Unitesi (YBU) klinik karar destek asistanisin.
Asagida bir ilac dozu izleme sisteminin tespit ettigi bir anomaliye ait veriler var.
Bu verileri degerlendirip KISA (en fazla 3-4 cumle), Turkce bir klinik yorum yaz.

Ilac: {drug_name} ({dose:.4f} {dose_unit})
Anomali zamani: {anomaly_time}
Rolling Z-Score: {z_score_text}

En onemli 3 SHAP ozelligi (tahmine katkisi buyuklugune gore):
{shap_lines}

Hasta vital bulgulari:
{vital_lines}

Yanitini yaklasik su formatta ver (kose parantezleri doldurup kaldir):
"Saat [X]'te anomali tespit edildi. SHAP analizi [ozellik] ozelliginin ana katki
sagladigini gosteriyor. Onerilen eylem: [eylem]."

Yalnizca bu klinik yorumu yaz, baska aciklama ekleme. Bu yorum bir karar destek
onerisidir; nihai klinik karar sorumlu hekime aittir."""

    return _call_claude(prompt, api_key=api_key, model=model, max_tokens=max_tokens)


def _call_claude(prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    """Claude Messages API'sini cagirir ve donen ilk metin blogunu dondurur.

    generate_clinical_commentary ve analyze_recent_data_with_claude arasinda
    paylasilan ortak hata isleme (kimlik dogrulama, hiz siniri, sunucu/aglantı
    hatasi, icerik politikasi reddi) burada tutulur.

    Args:
        prompt: Kullanici mesaji olarak gonderilecek tam prompt metni.
        api_key: Anthropic API anahtari.
        model: Kullanilacak Claude modeli.
        max_tokens: Yanit icin azami token sayisi.

    Returns:
        Yanitin ilk metin blogu (bastaki/sondaki bosluklar temizlenmis).

    Raises:
        RuntimeError: API cagrisi basarisiz olursa ya da Claude istegi
            icerik politikasi geregi reddederse, aciklayici Turkce mesajla.
    """
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError("Claude API kimlik dogrulama hatasi: API anahtarini kontrol edin.") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError("Claude API hiz siniri asildi, lutfen birazdan tekrar deneyin.") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API hatasi ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("Claude API'ye baglanilamadi, internet baglantisini kontrol edin.") from e

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude bu istegi icerik politikasi geregi yanitlamadi.")

    return next((b.text for b in response.content if b.type == "text"), "").strip()


# --------------------------------------------------------------------------
# Claude API ile ham veri analizi (anomali + trend + oneri) - matematik yok
# --------------------------------------------------------------------------

def analyze_recent_data_with_claude(
    window_df: pd.DataFrame,
    drug_columns: list[str],
    vital_columns: list[str],
    api_key: str,
    model: str = AI_ANALYSIS_MODEL,
    max_tokens: int = AI_ANALYSIS_MAX_TOKENS,
) -> dict:
    """Ham saatlik ilac dozu + vital bulgu verisini oldugu gibi Claude'a gonderip
    anomali tespiti, trend degerlendirmesi ve klinik oneriyi Claude'a yaptirir.

    Bu fonksiyon HICBIR istatistiksel hesaplama (Z-score, esik, model, vb.)
    yapmaz; anomali/trend/oneri degerlendirmesinin tamami Claude'un ham
    sayilari okuyarak yaptigi akil yurutmeye dayanir.

    Args:
        window_df: Analiz edilecek zaman penceresi (orn. son 24 saat).
            DatetimeIndex'li olmalidir.
        drug_columns: Gonderilecek ilac dozu sutunlari (orn. ["norepinephrine", "propofol"]).
        vital_columns: Gonderilecek vital bulgu sutunlari (orn. ["map", "heart_rate", "spo2", "lactate"]).
            window_df'de bulunmayan sutunlar sessizce atlanir.
        api_key: Anthropic API anahtari.
        model: Kullanilacak Claude modeli.
        max_tokens: Yanit icin azami token sayisi.

    Returns:
        dict: {
            "anomaly_detected": bool,
            "anomaly_hours": list[pd.Timestamp],  # Claude'un anomali olarak isaretledigi saatler
            "anomaly_explanation": str,           # (a) sorusunun Turkce yaniti
            "trend": str,                          # "yukseliyor" | "dusuyor" | "stabil" | "karisik"
            "trend_explanation": str,             # (b) sorusunun Turkce yaniti
            "recommendation": str,                # (c) sorusunun Turkce yaniti
            "raw_response": str,                  # Claude'un ham (JSON) yaniti
        }

    Raises:
        RuntimeError: API cagrisi basarisiz olursa (bkz. _call_claude).
        ValueError: Claude'un yaniti gecerli JSON olarak ayristirilamazsa.
    """
    available_drug_cols = [c for c in drug_columns if c in window_df.columns]
    available_vital_cols = [c for c in vital_columns if c in window_df.columns]
    all_cols = available_drug_cols + available_vital_cols

    lines = [f"saat | {' | '.join(all_cols)}"]
    for ts, row in window_df.iterrows():
        values = " | ".join(f"{row[c]:.3f}" for c in all_cols)
        lines.append(f"{ts} | {values}")
    data_text = "\n".join(lines)

    prompt = f"""Sen bir Yogun Bakim Unitesi (YBU) klinik veri analistisin. Asagida
saatlik ham hasta verisi var (ilac dozlari ve vital bulgular). Herhangi bir
istatistiksel hesaplama (Z-score, ortalama, esik vb.) YAPMADAN, YALNIZCA bu
sayilari inceleyerek kendi klinik degerlendirmeni yap.

Veri (saat | {' | '.join(all_cols)}):
{data_text}

Su 3 soruyu yanitla:
a) Anomali var mi? Varsa hangi saat(ler)de? (ilac dozunda veya vital bulguda
   ani/beklenmedik bir sapma)
b) Genel trend nasil? (yukseliyor / dusuyor / stabil / karisik)
c) Klinisyene oneri: ne yapilmali?

Yanitini SADECE asagidaki JSON formatinda ver; baska hicbir metin ekleme,
kod blogu (```) kullanma:

{{
  "anomali_var_mi": true veya false,
  "anomali_saatleri": ["yukaridaki veride kullanilan saat stringlerinden AYNEN kopyalanmis liste"],
  "anomali_aciklamasi": "a) sorusuna 1-3 cumlelik Turkce yanit",
  "trend": "yukseliyor" veya "dusuyor" veya "stabil" veya "karisik",
  "trend_aciklamasi": "b) sorusuna 1-3 cumlelik Turkce yanit",
  "oneri": "c) sorusuna 1-3 cumlelik Turkce yanit"
}}

Bu degerlendirme bir karar destek onerisidir; nihai klinik karar sorumlu hekime aittir."""

    raw_text = _call_claude(prompt, api_key=api_key, model=model, max_tokens=max_tokens)

    json_text = raw_text.strip()
    if json_text.startswith("```"):
        json_text = json_text.strip("`")
        if json_text.lower().startswith("json"):
            json_text = json_text[4:]
        json_text = json_text.strip()

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude yaniti gecerli JSON degil: {raw_text[:300]}") from e

    anomaly_hours = []
    for h in parsed.get("anomali_saatleri") or []:
        try:
            anomaly_hours.append(pd.Timestamp(h))
        except (ValueError, TypeError):
            continue

    return {
        "anomaly_detected": bool(parsed.get("anomali_var_mi", False)),
        "anomaly_hours": anomaly_hours,
        "anomaly_explanation": str(parsed.get("anomali_aciklamasi", "")),
        "trend": str(parsed.get("trend", "")),
        "trend_explanation": str(parsed.get("trend_aciklamasi", "")),
        "recommendation": str(parsed.get("oneri", "")),
        "raw_response": raw_text,
    }


if __name__ == "__main__":
    from anomaly import _inject_anomalies, classification_metrics, rolling_zscore_detect
    from data_manager import simulate_patient_data

    print("=== explain.py bagimsiz test ===")

    df = simulate_patient_data(n_hours=72, seed=42)
    df_anom, true_labels = _inject_anomalies(df, column="norepinephrine", n_anomalies=6, magnitude=5.0)
    series = df_anom["norepinephrine"]

    feat_df = build_feature_frame(series)
    model, feature_names, test_metrics = train_surrogate_model(feat_df)
    X_all = feat_df[feature_names].to_numpy()

    print(f"\nOzellikler: {feature_names}")
    print(f"Egitim orneklem sayisi: {len(feat_df)}")
    if "warning" in test_metrics:
        print(f"Vekil model test metrikleri: {test_metrics['warning']}")
    else:
        print(
            f"Vekil model out-of-sample (held-out) metrikleri: "
            f"MAE={test_metrics['mae']:.5f}  RMSE={test_metrics['rmse']:.5f}  "
            f"(egitim n={test_metrics['n_train']}, test n={test_metrics['n_test']})"
        )

    # --- Son tahmin icin KernelSHAP ve LIME ---
    x_last = X_all[-1]
    shap_res = kernelshap_explain(model, X_all, x_last, feature_names)
    lime_res = lime_explain(model, X_all, x_last, feature_names)

    print(f"\n--- Son gozlem icin tahmin: {shap_res['prediction']:.4f} ---")
    print("KernelSHAP katkilari (buyukluge gore):")
    for feat, val in shap_res["contributions"].items():
        print(f"  {feat:15s}: {val:+.5f}")

    print("\nLIME katkilari (buyukluge gore):")
    for feat, val in lime_res["contributions"].items():
        print(f"  {feat:15s}: {val:+.5f}")

    # --- Kural tabanli klinik yorum (API anahtari gerektirmez) ---
    print("\n=== generate_rule_based_commentary() testi ===")
    top3 = list(shap_res["contributions"].items())[:3]

    commentary_above = generate_rule_based_commentary(
        drug_name="norepinephrine", dose=float(series.iloc[-1]), dose_unit="mcg/kg/dk",
        anomaly_time=str(series.index[-1]), z_score=3.4, top_shap_features=top3,
        vitals={"MAP": 58.0, "HR": 112.0, "Laktat": 3.2},
    )
    print("(artis senaryosu, dusuk MAP + yuksek HR + yuksek laktat)")
    print(commentary_above)

    commentary_below = generate_rule_based_commentary(
        drug_name="insulin", dose=3.1, dose_unit="unite/saat",
        anomaly_time=str(series.index[-1]), z_score=-2.7, top_shap_features=top3,
        vitals={"HR": 52.0},
    )
    print("\n(azalis senaryosu, bradikardi)")
    print(commentary_below)

    # --- Anomali vs normal SHAP karsilastirmasi ---
    anomaly_positions = np.where(true_labels)[0]
    feat_index_map = {ts: i for i, ts in enumerate(feat_df.index)}
    anomaly_ts_candidates = [df_anom.index[p] for p in anomaly_positions if df_anom.index[p] in feat_index_map]

    if anomaly_ts_candidates:
        anomaly_ts = anomaly_ts_candidates[0]
        normal_ts = feat_df.index[5]  # erken, sakin bir donemden normal referans nokta

        x_anomaly = feat_df.loc[anomaly_ts, feature_names].to_numpy()
        x_normal = feat_df.loc[normal_ts, feature_names].to_numpy()

        comparison = compare_anomaly_vs_normal(model, X_all, x_anomaly, x_normal, feature_names)
        print(f"\n--- Anomali ({anomaly_ts}) vs Normal ({normal_ts}) SHAP karsilastirmasi ---")
        print(f"Tahmin (anomali):  {comparison.attrs['prediction_anomaly']:.4f}")
        print(f"Tahmin (normal):   {comparison.attrs['prediction_normal']:.4f}")
        print(comparison.to_string(index=False))
    else:
        print("\nUyari: ozellik penceresine dusen enjekte edilmis anomali bulunamadi (dropna sinirlari).")

    print("\n=== generate_clinical_commentary() testi (Claude API) ===")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Atlaniyor - ANTHROPIC_API_KEY ortam degiskeni tanimli degil.")
    else:
        try:
            top3 = list(shap_res["contributions"].items())[:3]
            commentary = generate_clinical_commentary(
                drug_name="norepinephrine",
                dose=float(series.iloc[-1]),
                dose_unit="mcg/kg/dk",
                anomaly_time=str(series.index[-1]),
                z_score=2.8,
                top_shap_features=top3,
                vitals={"MAP": 65.0, "HR": 110.0, "Laktat": 3.2},
                api_key=api_key,
            )
            print(commentary)
        except RuntimeError as e:
            print(f"Hata: {e}")

    print("\n=== analyze_recent_data_with_claude() testi (Claude API) ===")
    if not api_key:
        print("Atlaniyor - ANTHROPIC_API_KEY ortam degiskeni tanimli degil.")
    else:
        try:
            window_df = df_anom.tail(24)
            result = analyze_recent_data_with_claude(
                window_df=window_df,
                drug_columns=["norepinephrine", "propofol"],
                vital_columns=["heart_rate", "map", "spo2"],
                api_key=api_key,
            )
            print("Anomali var mi:", result["anomaly_detected"], "- saatler:", result["anomaly_hours"])
            print("Aciklama:", result["anomaly_explanation"])
            print("Trend:", result["trend"], "-", result["trend_explanation"])
            print("Oneri:", result["recommendation"])
        except (RuntimeError, ValueError) as e:
            print(f"Hata: {e}")
