"""
app_streamlit.py
----------------
YBU (Yogun Bakim Unitesi) ilac dozu izleme sistemi icin Streamlit paneli.

Bu uygulama, data_manager / stat_analysis / forecast / anomaly / explain
modullerini tek bir interaktif akiskta birlestirir:

    1. Simule hasta verisi uretimi (norepinefrin, propofol, insulin, heparin)
    2. Durganlik analizi (ADF / KPSS)
    3. ARIMA + LSTM hibrit tahmin ve walk-forward backtest
    4. Rolling Z-Score + Isolation Forest + LSTM Autoencoder anomali ensemble
    5. KernelSHAP / LIME ile son tahmin ve anomali aciklamasi

Calistirma:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from anomaly import (
    classification_metrics,
    ensemble_detect,
    isolation_forest_detect,
    isolation_forest_features_for_drug,
    lstm_autoencoder_detect,
    rolling_zscore_detect,
    rolling_zscore_values,
)
from data_manager import (
    DRUGS,
    VITALS,
    compute_log_returns,
    fill_missing,
    inject_missing,
    list_mimic_stays,
    load_mimic_data,
    simulate_patient_data,
)
from explain import (
    analyze_recent_data_with_claude,
    build_feature_frame,
    compare_anomaly_vs_normal,
    generate_rule_based_commentary,
    kernelshap_explain,
    lime_explain,
    train_surrogate_model,
)
from forecast import compute_metrics, forecast_arima, forecast_lstm, fit_arima, hybrid_forecast, rmse, train_lstm, walk_forward_backtest
from stat_analysis import stationarity_report

st.set_page_config(page_title="YBU Ilac Dozu Izleme Sistemi", layout="wide")

DRUG_UNITS = {
    "norepinephrine": "mcg/kg/dk",
    "propofol": "mcg/kg/dk",
    "insulin": "unite/saat",
    "heparin": "unite/saat",
}


@st.cache_data(show_spinner=False)
def get_data(n_hours: int, seed: int, missing_frac: float) -> pd.DataFrame:
    raw = simulate_patient_data(n_hours=n_hours, seed=seed)
    if missing_frac > 0:
        raw = inject_missing(raw, frac=missing_frac, seed=seed)
    return fill_missing(raw)


@st.cache_data(show_spinner=False)
def get_mimic_stays(data_dir: str) -> pd.DataFrame:
    return list_mimic_stays(data_dir=data_dir)


@st.cache_data(show_spinner=False)
def get_mimic_patient_data(data_dir: str, stay_id: int) -> pd.DataFrame:
    return load_mimic_data(data_dir=data_dir, stay_id=stay_id)


@st.cache_data(show_spinner=False)
def get_anomaly_masks(df: pd.DataFrame, drug: str, contamination: float) -> dict:
    series = df[drug]
    z_mask = rolling_zscore_detect(series)
    if_features = isolation_forest_features_for_drug(drug, df.columns.tolist())
    if_mask = isolation_forest_detect(df, features=if_features, contamination=contamination)
    ae_mask = lstm_autoencoder_detect(series, epochs=100)
    majority = ensemble_detect({"zscore": z_mask, "isolation_forest": if_mask, "autoencoder": ae_mask}, method="majority")
    union = ensemble_detect({"zscore": z_mask, "isolation_forest": if_mask, "autoencoder": ae_mask}, method="union")
    return {
        "Rolling Z-Score": z_mask,
        "Isolation Forest": if_mask,
        "LSTM Autoencoder": ae_mask,
        "Ensemble (coğunluk)": majority,
        "Ensemble (birlesim)": union,
    }


@st.cache_data(show_spinner=False)
def get_backtest(series: np.ndarray, initial_window: int, horizon: int, step: int, lstm_epochs: int) -> dict:
    return walk_forward_backtest(series, initial_window=initial_window, horizon=horizon, step=step, lstm_epochs=lstm_epochs)


@st.cache_data(show_spinner=False)
def get_last_point_forecast(series: np.ndarray, test_len: int, lstm_epochs: int) -> dict:
    train, test = series[:-test_len], series[-test_len:]
    arima_fitted = fit_arima(train)
    arima_pred = forecast_arima(arima_fitted, steps=test_len)
    lstm_model, scaler, epoch_losses = train_lstm(train, epochs=lstm_epochs)
    lstm_pred = forecast_lstm(lstm_model, scaler, train, steps=test_len)
    arima_err = rmse(test, arima_pred)
    lstm_err = rmse(test, lstm_pred)
    hybrid_pred = hybrid_forecast(arima_pred, lstm_pred, arima_err, lstm_err)
    return {
        "test": test,
        "arima_pred": arima_pred,
        "lstm_pred": lstm_pred,
        "hybrid_pred": hybrid_pred,
        "epoch_losses": epoch_losses,
        "metrics": {
            "ARIMA": compute_metrics(test, arima_pred),
            "LSTM": compute_metrics(test, lstm_pred),
            "Hibrit": compute_metrics(test, hybrid_pred),
        },
    }


# --------------------------------------------------------------------------
# Kenar cubugu (sidebar) - kontroller
# --------------------------------------------------------------------------

st.sidebar.title("Kontrol Paneli")

veri_kaynagi = st.sidebar.radio(
    "Veri Kaynagi",
    ["Simule Veri", "MIMIC-IV Gercek Veri"],
)

st.sidebar.markdown("---")

df = None
data_source_caption = ""
MIMIC_DATA_DIR = "mimic_data"

if veri_kaynagi == "Simule Veri":
    n_hours = st.sidebar.slider("Simulasyon suresi (saat)", min_value=48, max_value=168, value=72, step=12)
    seed = st.sidebar.number_input("Rastgelelik tohumu (seed)", min_value=0, max_value=9999, value=42)
    missing_frac = st.sidebar.slider("Eksik veri orani", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
    df = get_data(n_hours, seed, missing_frac)
    data_source_caption = f"Simule veri ({n_hours} saat, seed={seed})"

else:
    st.sidebar.subheader("MIMIC-IV Hasta Secimi")
    try:
        stays = get_mimic_stays(MIMIC_DATA_DIR)
    except FileNotFoundError:
        st.error(
            f"MIMIC-IV verisi bulunamadi: `{MIMIC_DATA_DIR}/` klasorunde icustays.csv "
            "(veya .csv.gz) yok. Lutfen MIMIC-IV dosyalarini bu klasore yerlestirin "
            "ya da 'Simule Veri' secenegini kullanin."
        )
        stays = None
    except Exception as e:
        st.error(f"MIMIC-IV yatis listesi okunurken beklenmeyen bir hata olustu: {e}")
        stays = None

    if stays is not None:
        if len(stays) == 0:
            st.sidebar.warning(f"{MIMIC_DATA_DIR}/icustays.csv icinde hic ICU yatisi bulunamadi.")
        else:
            stay_options = {
                f"{row.stay_id}  (hasta={row.subject_id}, ilac sayisi={row.n_drugs}, giris={row.intime.date()})": int(row.stay_id)
                for row in stays.itertuples()
            }
            selected_label = st.sidebar.selectbox("ICU Yatisi (stay_id)", list(stay_options.keys()))
            selected_stay_id = stay_options[selected_label]
            st.sidebar.caption(
                "Ilac sayisi=0 olan yatislarda norepinefrin/propofol/insulin/heparin "
                "kayitlarindan hicbiri yoktur; secilirse veri bos gorunebilir."
            )

            try:
                df = get_mimic_patient_data(MIMIC_DATA_DIR, selected_stay_id)
                data_source_caption = f"MIMIC-IV gercek veri (stay_id={selected_stay_id})"
            except FileNotFoundError as e:
                st.error(f"MIMIC-IV dosyalari okunamadi: {e}")
            except ValueError as e:
                st.error(f"Secilen yatis (stay_id={selected_stay_id}) icin veri olusturulamadi: {e}")
            except Exception as e:
                st.error(f"Hasta verisi yuklenirken beklenmeyen bir hata olustu: {e}")

if df is None:
    st.title("YBU Ilac Dozu Izleme Sistemi")
    st.error("Veri yuklenemedi. Lutfen kenar cubugundan bir veri kaynagi/hasta secin.")
    st.stop()

drug = st.sidebar.selectbox("Izlenecek ilac", DRUGS, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Anomali tespiti")
contamination = st.sidebar.slider("Beklenen anomali orani", min_value=0.01, max_value=0.15, value=0.05, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Tahmin / Backtest")
test_len = st.sidebar.slider("Tek seferlik test ufku (saat)", min_value=6, max_value=24, value=12, step=2)
initial_window = st.sidebar.slider("Backtest baslangic penceresi (saat)", min_value=24, max_value=60, value=36, step=6)
backtest_step = st.sidebar.slider("Backtest adim buyuklugu (saat)", min_value=3, max_value=12, value=6, step=3)
lstm_epochs = st.sidebar.slider("LSTM epoch sayisi", min_value=10, max_value=100, value=40, step=10)

run_backtest = st.sidebar.checkbox("Walk-forward backtest calistir (yavas)", value=False)

st.title("YBU Ilac Dozu Izleme Sistemi")
st.caption(f"Veri kaynagi: {data_source_caption}. Durganlik analizi, hibrit ARIMA+LSTM tahmini, anomali ensemble ve SHAP/LIME acikanabilirligi.")

series = df[drug]

tabs = st.tabs(["Veri", "Durganlik", "Tahmin", "Anomali", "Aciklanabilirlik", "AI Analizi"])


# --------------------------------------------------------------------------
# Tab 1: Veri
# --------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Simule Hasta Verisi")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(df[list(DRUGS)])
        st.line_chart(df[list(VITALS)])
    with col2:
        st.metric(f"{drug} - son deger", f"{series.iloc[-1]:.3f} {DRUG_UNITS[drug]}")
        st.metric(f"{drug} - ortalama", f"{series.mean():.3f} {DRUG_UNITS[drug]}")
        st.metric(f"{drug} - std", f"{series.std():.3f}")
        st.dataframe(df.describe().T[["mean", "std", "min", "max"]], use_container_width=True)

    st.subheader("Ham veri")
    st.dataframe(df, use_container_width=True, height=250)

    log_returns = compute_log_returns(df, columns=list(DRUGS))
    st.subheader("Log-getiri (ilaclar)")
    st.line_chart(log_returns.dropna())


# --------------------------------------------------------------------------
# Tab 2: Durganlik
# --------------------------------------------------------------------------

with tabs[1]:
    st.subheader(f"{drug} icin Durganlik Analizi (ADF / KPSS)")

    raw_report = stationarity_report(series)
    log_series = compute_log_returns(df, columns=[drug])[f"{drug}_log_return"].dropna()
    log_report = stationarity_report(log_series)

    col1, col2 = st.columns(2)
    for col, title, report in [(col1, "Ham doz serisi", raw_report), (col2, "Log-getiri serisi", log_report)]:
        with col:
            st.markdown(f"**{title}**")
            st.write(f"ADF p-degeri: `{report['adf']['p_value']:.4f}` -> durgan: **{report['adf']['is_stationary']}**")
            st.write(f"KPSS p-degeri: `{report['kpss']['p_value']:.4f}` -> durgan: **{report['kpss']['is_stationary']}**")
            st.info(f"Sonuc: **{report['verdict']}**")

    st.caption(
        "ADF H0: seri durgan degil (p<0.05 ise durgan). "
        "KPSS H0: seri durgan (p<0.05 ise durgan degil)."
    )


# --------------------------------------------------------------------------
# Tab 3: Tahmin
# --------------------------------------------------------------------------

with tabs[2]:
    st.subheader(f"{drug} icin ARIMA(2,0,1) + LSTM Hibrit Tahmin")

    with st.spinner("ARIMA ve LSTM modelleri egitiliyor..."):
        result = get_last_point_forecast(series.to_numpy(), test_len, lstm_epochs)

    plot_df = pd.DataFrame({
        "Gercek": result["test"],
        "ARIMA": result["arima_pred"],
        "LSTM": result["lstm_pred"],
        "Hibrit": result["hybrid_pred"],
    }, index=series.index[-test_len:])
    st.line_chart(plot_df)

    metrics_df = pd.DataFrame(result["metrics"]).T
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("LSTM Ogrenme Egrisi")
    epoch_losses = result["epoch_losses"]
    loss_df = pd.DataFrame(
        {"MSE Kaybi": epoch_losses},
        index=pd.RangeIndex(1, len(epoch_losses) + 1, name="Epoch"),
    )
    st.line_chart(loss_df)
    st.caption("Egri asagiya dogru gidiyorsa model dogru ogreniyor.")

    if run_backtest:
        st.subheader("Walk-Forward Backtest")
        with st.spinner("Backtest calisiyor (her pencerede ARIMA + LSTM yeniden egitiliyor)..."):
            bt = get_backtest(series.to_numpy(), initial_window, 1, backtest_step, lstm_epochs)
        bt_metrics_df = pd.DataFrame(bt["metrics"]).T
        st.dataframe(bt_metrics_df.style.format("{:.4f}"), use_container_width=True)

        bt_plot_df = pd.DataFrame({
            "Gercek": bt["y_true"],
            "ARIMA": bt["arima_pred"],
            "LSTM": bt["lstm_pred"],
            "Hibrit": bt["hybrid_pred"],
        })
        st.line_chart(bt_plot_df)
    else:
        st.caption("Backtest varsayilan olarak kapali (yavas calisir). Kenar cubugundan acabilirsiniz.")


# --------------------------------------------------------------------------
# Tab 4: Anomali
# --------------------------------------------------------------------------

with tabs[3]:
    st.subheader(f"{drug} icin Anomali Tespiti (Ensemble)")

    with st.spinner("Rolling Z-Score, Isolation Forest ve LSTM Autoencoder calisiyor..."):
        masks = get_anomaly_masks(df, drug, contamination)

    method = st.radio("Goruntulenecek yontem", list(masks.keys()), horizontal=True, index=3)
    mask = masks[method]

    chart_df = pd.DataFrame({drug: series})
    st.line_chart(chart_df)

    anomaly_points = series[mask]
    st.write(f"**{method}** -> tespit edilen anomali sayisi: {int(mask.sum())}")
    if len(anomaly_points) > 0:
        st.dataframe(anomaly_points.rename("deger").to_frame(), use_container_width=True)
    else:
        st.caption("Bu yontemle anomali tespit edilmedi.")

    st.markdown("---")
    st.markdown("**Tum yontemlerin karsilastirmasi (tespit sayisi)**")
    counts = {name: int(m.sum()) for name, m in masks.items()}
    st.bar_chart(pd.Series(counts, name="tespit sayisi"))

    st.caption(
        "Not: Panelde 'gercek' anomali etiketi bulunmadigindan (simule veri organik gurultu icerir), "
        "F1/Precision/Recall hesaplamasi icin `python anomaly.py` bagimsiz testine bakiniz "
        "(orada bilinen anomaliler enjekte edilip metrikler raporlanir)."
    )

    st.markdown("---")
    st.subheader("Klinik Yorum (Kural Tabanli)")
    st.caption(
        "Secilen anomali noktasi icin Z-Score, en onemli SHAP ozelligi ve vital bulgulara "
        "dayali, if/else mantigiyla uretilen Turkce klinik yorum. Claude API veya "
        "ANTHROPIC_API_KEY gerektirmez."
    )

    if len(anomaly_points) == 0:
        st.caption("Klinik yorum uretmek icin once secilen yontemle en az bir anomali tespit edilmelidir.")
    else:
        anomaly_ts_options = anomaly_points.index.tolist()
        selected_anomaly_ts = st.selectbox(
            "Yorumlanacak anomali zaman noktasi", anomaly_ts_options, key="clinical_comment_ts"
        )

        if st.button("Klinik yorum olustur"):
            feat_df = build_feature_frame(series)
            if selected_anomaly_ts not in feat_df.index:
                st.warning(
                    "Bu zaman noktasi ozellik penceresi icin yeterli gecmis veriye sahip degil "
                    "(serinin ilk birkac saati)."
                )
            else:
                with st.spinner("Kural tabanli klinik yorum olusturuluyor..."):
                    surrogate_model, feature_names, _surrogate_metrics = train_surrogate_model(feat_df)
                    X_all = feat_df[feature_names].to_numpy()
                    x_point = feat_df.loc[selected_anomaly_ts, feature_names].to_numpy()

                    shap_res_point = kernelshap_explain(surrogate_model, X_all, x_point, feature_names)
                    top3 = list(shap_res_point["contributions"].items())[:3]

                    z_series = rolling_zscore_values(series)
                    z_value = z_series.get(selected_anomaly_ts)
                    z_value = float(z_value) if pd.notna(z_value) else None

                    vitals = {}
                    for col, label in [("map", "MAP"), ("heart_rate", "HR"), ("lactate", "Laktat")]:
                        if col in df.columns and pd.notna(df.loc[selected_anomaly_ts, col]):
                            vitals[label] = float(df.loc[selected_anomaly_ts, col])

                    commentary = generate_rule_based_commentary(
                        drug_name=drug,
                        dose=float(series.loc[selected_anomaly_ts]),
                        dose_unit=DRUG_UNITS[drug],
                        anomaly_time=str(selected_anomaly_ts),
                        z_score=z_value,
                        top_shap_features=top3,
                        vitals=vitals,
                    )
                    st.info(commentary)


# --------------------------------------------------------------------------
# Tab 5: Aciklanabilirlik
# --------------------------------------------------------------------------

with tabs[4]:
    st.subheader(f"{drug} icin KernelSHAP / LIME Aciklamasi")

    feat_df = build_feature_frame(series)
    if len(feat_df) < 20:
        st.warning("Aciklanabilirlik icin yeterli veri yok, simulasyon suresini artirin.")
    else:
        model, feature_names, surrogate_metrics = train_surrogate_model(feat_df)
        X_all = feat_df[feature_names].to_numpy()
        x_last = X_all[-1]

        with st.spinner("KernelSHAP ve LIME hesaplaniyor..."):
            shap_res = kernelshap_explain(model, X_all, x_last, feature_names)
            lime_res = lime_explain(model, X_all, x_last, feature_names)

        m1, m2, m3 = st.columns(3)
        m1.metric("Son gozlem icin vekil model tahmini", f"{shap_res['prediction']:.4f}")
        if "warning" in surrogate_metrics:
            st.caption(f"Vekil model dogrulugu: {surrogate_metrics['warning']}")
        else:
            m2.metric("Vekil model out-of-sample MAE", f"{surrogate_metrics['mae']:.4f}")
            m3.metric("Vekil model out-of-sample RMSE", f"{surrogate_metrics['rmse']:.4f}")
            st.caption(
                f"MAE/RMSE, modelin GORMEDIGI son {surrogate_metrics['n_test']} gozlemde "
                f"(egitim n={surrogate_metrics['n_train']}) hesaplanmistir - in-sample degil, "
                "gercek bir tutulmus-veri (held-out) degerlendirmesidir."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KernelSHAP katkilari**")
            st.bar_chart(pd.Series(shap_res["contributions"], name="shap_degeri"))
        with col2:
            st.markdown("**LIME katkilari**")
            st.bar_chart(pd.Series(lime_res["contributions"], name="lime_agirligi"))

        st.markdown("---")
        st.markdown("**Anomali vs Normal SHAP karsilastirmasi**")

        masks = get_anomaly_masks(df, drug, contamination)
        anomaly_mask = masks["Ensemble (birlesim)"]
        anomaly_ts_candidates = [ts for ts in feat_df.index if anomaly_mask.get(ts, False)]

        if anomaly_ts_candidates:
            anomaly_ts = st.selectbox("Anomali zaman noktasi", anomaly_ts_candidates)
            normal_ts = feat_df.index[len(feat_df) // 4]

            x_anomaly = feat_df.loc[anomaly_ts, feature_names].to_numpy()
            x_normal = feat_df.loc[normal_ts, feature_names].to_numpy()

            with st.spinner("Karsilastirmali SHAP hesaplaniyor..."):
                comparison = compare_anomaly_vs_normal(model, X_all, x_anomaly, x_normal, feature_names)

            c1, c2 = st.columns(2)
            c1.metric(f"Tahmin (anomali @ {anomaly_ts})", f"{comparison.attrs['prediction_anomaly']:.4f}")
            c2.metric(f"Tahmin (normal @ {normal_ts})", f"{comparison.attrs['prediction_normal']:.4f}")

            st.dataframe(comparison.style.format({"shap_anomaly": "{:.4f}", "shap_normal": "{:.4f}", "abs_diff": "{:.4f}"}),
                         use_container_width=True)
        else:
            st.caption("Mevcut ensemble ayarlariyla ozellik penceresine dusen bir anomali bulunamadi.")


# --------------------------------------------------------------------------
# Tab 6: AI Analizi (yalnizca Claude API - matematik/istatistik yok)
# --------------------------------------------------------------------------

AI_DRUG_COLUMNS = ["norepinephrine", "propofol"]
AI_VITAL_COLUMNS = ["map", "heart_rate", "spo2", "lactate"]

with tabs[5]:
    st.subheader("AI Analizi (Claude API)")
    st.caption(
        "Bu sekme diger sekmelerden bagimsiz calisir. Son 24 saatin ham ilac dozu ve "
        "vital bulgu verisi dogrudan Claude API'ye gonderilir; anomali tespiti, trend "
        "degerlendirmesi ve klinik oneri TAMAMEN Claude tarafindan, herhangi bir "
        "istatistiksel hesaplama yapilmadan uretilir."
    )

    window_df = df.tail(24)
    missing_vitals = [c for c in AI_VITAL_COLUMNS if c not in df.columns]

    st.write(f"Analiz penceresi: son {len(window_df)} saat ({window_df.index.min()} -> {window_df.index.max()})")
    if missing_vitals:
        st.caption(f"Not: {', '.join(missing_vitals)} bu veri kaynaginda mevcut degil, gonderilmeyecek.")

    try:
        ai_api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        ai_api_key = None

    if not ai_api_key:
        st.warning(
            "ANTHROPIC_API_KEY bulunamadi. `.streamlit/secrets.toml` dosyasina "
            '`ANTHROPIC_API_KEY = "sk-ant-..."` satirini ekleyin.'
        )
    elif st.button("Claude ile analiz et"):
        with st.spinner("Claude son 24 saatlik veriyi degerlendiriyor..."):
            try:
                result = analyze_recent_data_with_claude(
                    window_df=window_df,
                    drug_columns=AI_DRUG_COLUMNS,
                    vital_columns=AI_VITAL_COLUMNS,
                    api_key=ai_api_key,
                )
            except (RuntimeError, ValueError) as e:
                st.error(str(e))
                result = None

        if result is not None:
            st.markdown("**a) Anomali var mi?**")
            st.info(result["anomaly_explanation"] or "Claude bir aciklama dondurmedi.")

            st.markdown("**b) Trend nasil?**")
            trend_label = result["trend"] or "belirtilmedi"
            st.info(f"({trend_label}) {result['trend_explanation'] or 'Claude bir aciklama dondurmedi.'}")

            st.markdown("**c) Oneri: klinisyen ne yapmali?**")
            st.info(result["recommendation"] or "Claude bir oneri dondurmedi.")

            st.markdown("---")
            st.markdown("**Algoritma vs Claude Karsilastirmasi**")
            st.caption(
                "Algoritma sayisi, ayni pencere icin mevcut Rolling Z-Score + Isolation "
                "Forest + LSTM Autoencoder ensemble'inin (birlesim) isaretledigi farkli "
                "saat sayisidir; Claude'un kendi analizinde kullanilmaz, yalnizca "
                "karsilastirma amaciyla ayrica hesaplanir."
            )

            with st.spinner("Karsilastirma icin algoritmik anomali tespiti calisiyor..."):
                algo_hours = set()
                for col in AI_DRUG_COLUMNS:
                    if col not in df.columns:
                        continue
                    col_series = df[col]
                    z_mask = rolling_zscore_detect(col_series)
                    if_features = isolation_forest_features_for_drug(col, df.columns.tolist())
                    if_mask = isolation_forest_detect(df, features=if_features, contamination=0.05)
                    ae_mask = lstm_autoencoder_detect(col_series, epochs=100)
                    union_mask = ensemble_detect(
                        {"zscore": z_mask, "isolation_forest": if_mask, "autoencoder": ae_mask},
                        method="union",
                    )
                    flagged = union_mask[union_mask].index
                    algo_hours.update(ts for ts in flagged if ts in window_df.index)

            claude_hours = {pd.Timestamp(h).floor("h") for h in result["anomaly_hours"]}
            algo_hours_norm = {pd.Timestamp(h).floor("h") for h in algo_hours}

            union_size = len(claude_hours | algo_hours_norm)
            intersection_size = len(claude_hours & algo_hours_norm)
            match_pct = 100.0 if union_size == 0 else (intersection_size / union_size) * 100.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Algoritma buldu", f"{len(algo_hours_norm)} anomali")
            c2.metric("Claude buldu", f"{len(claude_hours)} anomali")
            c3.metric("Uyusma", f"%{match_pct:.0f}")

            with st.expander("Claude'un ham JSON yaniti"):
                st.code(result["raw_response"], language="json")
    else:
        st.caption("Analizi baslatmak icin yukaridaki butona tiklayin (API cagrisi icerdiginden otomatik calismaz).")
