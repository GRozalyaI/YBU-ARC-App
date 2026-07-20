"""
data_manager.py
---------------
YBU (Yogun Bakim Unitesi) ilac dozu izleme sistemi icin veri katmani.

Sorumluluklari:
    - Saatlik ilac dozu ve vital bulgu CSV dosyalarini yukleme
    - MIMIC-IV inputevents/chartevents/labevents dosyalarindan gercek
      hasta verisi cikarma (load_mimic_data)
    - Log-getiri (log-return) hesaplama
    - Eksik veri doldurma (interpolasyon + ffill/bfill)
    - Norepinefrin, propofol, insulin ve heparin icin 72 saatlik
      simule hasta verisi uretme

Bagimsiz calistirma:
    python data_manager.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DRUGS = ["norepinephrine", "propofol", "insulin", "heparin"]
VITALS = ["heart_rate", "map", "spo2", "glucose"]

# MIMIC-IV d_items / d_labitems itemid eslemeleri
_MIMIC_DRUG_ITEMIDS = {
    "norepinephrine": 221906,
    "propofol": 222168,
    "insulin": 223257,
    "heparin": 225975,
}
_MIMIC_VITAL_ITEMIDS = {
    "map": 220052,
    "heart_rate": 220045,
    "spo2": 220277,
}
_MIMIC_LAB_ITEMIDS = {
    "lactate": 50813,
    "glucose": 50931,
}

# Klinik olarak makul baz doz araliklari (birimler yorum satirinda)
_DRUG_BASELINE = {
    "norepinephrine": 0.08,   # mcg/kg/dk
    "propofol": 30.0,         # mcg/kg/dk
    "insulin": 4.0,           # unite/saat
    "heparin": 900.0,         # unite/saat
}
_DRUG_NOISE_STD = {
    "norepinephrine": 0.015,
    "propofol": 4.0,
    "insulin": 0.6,
    "heparin": 60.0,
}


def simulate_patient_data(n_hours: int = 72, seed: int | None = 42) -> pd.DataFrame:
    """72 saatlik (varsayilan) simule YBU hasta verisi uretir.

    Ilac dozlari icin ornekleme deseni:
        - Rastgele yuruyus (random walk) + periyodik bolus enjeksiyonlari
        - Fizyolojik olarak makul alt/ust sinir kirpma (clipping)

    Vital bulgular ilac dozlarina bagimli olarak (basit lineer + gurultu)
    uretilir, boylece ileride kurulacak modeller icin gercekci korelasyon
    saglanir:
        - norepinefrin dogru orantili -> ortalama arter basinci (MAP)
        - propofol ters orantili       -> kalp hizi (sedasyon derinligi)
        - insulin ters orantili        -> glukoz
        - heparin dogru orantili       -> kalp hizinda hafif degisim (gurultu araci)

    Args:
        n_hours: Uretilecek saatlik gozlem sayisi.
        seed: Tekrarlanabilirlik icin rastgelelik tohumu.

    Returns:
        DatetimeIndex'li (saatlik, 'h' frekans) pandas.DataFrame.
        Sutunlar: DRUGS + VITALS.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 00:00", periods=n_hours, freq="h")

    data: dict[str, np.ndarray] = {}

    for drug in DRUGS:
        baseline = _DRUG_BASELINE[drug]
        noise_std = _DRUG_NOISE_STD[drug]

        series = np.empty(n_hours)
        series[0] = baseline
        for t in range(1, n_hours):
            step = rng.normal(0, noise_std)
            # Klinisyenin doz titrasyonunu taklit eden ara sira bolus/kesinti
            if rng.random() < 0.05:
                step += rng.choice([-1, 1]) * noise_std * rng.uniform(3, 6)
            series[t] = series[t - 1] + step

        lower = baseline * 0.2
        upper = baseline * 3.0
        series = np.clip(series, lower, upper)
        data[drug] = series

    # Vital bulgular: baz deger + ilaca bagli katki + gurultu
    hr = 85 + (-0.5) * (data["propofol"] - _DRUG_BASELINE["propofol"]) \
         + rng.normal(0, 3, n_hours) + 5 * np.sin(np.linspace(0, 6 * np.pi, n_hours))
    mean_ap = 70 + 15 * (data["norepinephrine"] - _DRUG_BASELINE["norepinephrine"]) / _DRUG_BASELINE["norepinephrine"] \
         + rng.normal(0, 2.5, n_hours)
    spo2 = 97 + rng.normal(0, 0.8, n_hours) - 0.02 * (data["propofol"] - _DRUG_BASELINE["propofol"])
    glucose = 140 - 6 * (data["insulin"] - _DRUG_BASELINE["insulin"]) + rng.normal(0, 6, n_hours)

    data["heart_rate"] = np.clip(hr, 40, 180)
    data["map"] = np.clip(mean_ap, 40, 130)
    data["spo2"] = np.clip(spo2, 70, 100)
    data["glucose"] = np.clip(glucose, 50, 400)

    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "timestamp"
    return df


def inject_missing(df: pd.DataFrame, frac: float = 0.05, seed: int | None = 0) -> pd.DataFrame:
    """Test amacli, DataFrame'e rastgele NaN degerler enjekte eder.

    Args:
        df: Kaynak DataFrame.
        frac: Her sutunda NaN yapilacak hucre orani (0-1).
        seed: Rastgelelik tohumu.

    Returns:
        Eksik veri enjekte edilmis kopya DataFrame.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    for col in out.columns:
        n_missing = int(n * frac)
        idx = rng.choice(n, size=n_missing, replace=False)
        out.iloc[idx, out.columns.get_loc(col)] = np.nan
    return out


def fill_missing(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """Eksik degerleri doldurur.

    Args:
        df: Eksik deger icerebilen DataFrame.
        method: "interpolate" (zaman bazli lineer interpolasyon,
            ardindan ffill/bfill ile uc noktalari kapatir) ya da
            "ffill" (sadece ileri doldurma + geriye kalan bosluklar icin bfill).

    Returns:
        Eksik degerleri doldurulmus yeni DataFrame.
    """
    out = df.copy()
    if method == "interpolate":
        out = out.interpolate(method="time" if isinstance(out.index, pd.DatetimeIndex) else "linear")
        out = out.ffill().bfill()
    elif method == "ffill":
        out = out.ffill().bfill()
    else:
        raise ValueError(f"Bilinmeyen doldurma yontemi: {method}")
    return out


def compute_log_returns(df: pd.DataFrame, columns: list[str] | None = None, eps: float = 1e-6) -> pd.DataFrame:
    """Log-getiri hesaplar: r_t = log(x_t + eps) - log(x_{t-1} + eps).

    Ilac dozlari ve vital bulgular negatif olamayacagindan, sifira yakin
    degerlerde log(0) hatasini onlemek icin kucuk bir eps sabiti eklenir.

    Args:
        df: Kaynak DataFrame (eksik verisi olmamali, once fill_missing kullanin).
        columns: Log-getirisi hesaplanacak sutunlar; None ise tum sutunlar.
        eps: Sayisal kararlilik icin kucuk sabit.

    Returns:
        Ayni index'e sahip, secilen sutunlar icin log-getiri DataFrame'i.
        Ilk satir NaN olur (onceki gozlem olmadigindan).
    """
    cols = columns or list(df.columns)
    log_vals = np.log(df[cols] + eps)
    log_returns = log_vals.diff()
    log_returns.columns = [f"{c}_log_return" for c in cols]
    return log_returns


def load_from_csv(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """CSV'den saatlik ilac dozu / vital bulgu verisini yukler.

    Args:
        path: CSV dosya yolu.
        timestamp_col: Zaman damgasi sutununun adi.

    Returns:
        DatetimeIndex'li, zamana gore sirali pandas.DataFrame.
    """
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    df.index.name = "timestamp"
    return df


def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """DataFrame'i CSV'ye yazar (index dahil)."""
    df.to_csv(path, index=True, index_label="timestamp")


def _find_mimic_file(data_dir: str, table_name: str) -> str:
    """MIMIC-IV tablo dosyasini data_dir altinda (duz veya `icu/`, `hosp/` gibi
    ic klasorlerde) `.csv` ya da `.csv.gz` uzantisiyla arar.

    Resmi MIMIC-IV dagitiminda tablolar modullere gore alt klasorlere
    ayrilmis (orn. icu/inputevents.csv.gz, hosp/labevents.csv.gz) ve
    sikistirilmis (gzip) olarak gelir; bu fonksiyon dosyanin data_dir
    icinde tam olarak nerede oldugunu bilmeye gerek birakmadan bulur.

    Args:
        data_dir: Aranacak kok dizin.
        table_name: Uzantisiz tablo adi (orn. "inputevents").

    Returns:
        Bulunan dosyanin tam yolu.

    Raises:
        FileNotFoundError: Ne `.csv` ne de `.csv.gz` varyanti bulunamazsa.
    """
    from pathlib import Path

    matches = sorted(Path(data_dir).rglob(f"{table_name}.csv")) + \
        sorted(Path(data_dir).rglob(f"{table_name}.csv.gz"))
    if not matches:
        raise FileNotFoundError(
            f"MIMIC-IV dosyasi bulunamadi: {table_name}.csv / {table_name}.csv.gz "
            f"({data_dir} altinda, alt klasorler dahil aranmasina ragmen)"
        )
    return str(matches[0])


def _read_filtered_csv(path: str, itemids: list[int], usecols: list[str],
                        parse_dates: list[str], chunksize: int = 200_000) -> pd.DataFrame:
    """Buyuk MIMIC-IV CSV(.gz) dosyalarini (chartevents/labevents onlarca GB
    olabilir) tamamini belleğe yuklemeden, parca parca (chunked) okuyup
    sadece istenen itemid'lere ait satirlari tutar.

    `.csv.gz` dosyalar icin sikistirma, pandas tarafindan dosya uzantisindan
    otomatik olarak algilanir (compression="infer"); chunked okuma gzip
    akisinda da calisir.

    Args:
        path: CSV veya CSV.GZ dosya yolu.
        itemids: Tutulacak itemid degerleri.
        usecols: Okunacak sutunlar (bellek/hiz icin daraltilmis).
        parse_dates: Datetime'a cevrilecek sutunlar.
        chunksize: Her seferde okunacak satir sayisi.

    Returns:
        Filtrelenmis satirlari iceren pandas.DataFrame (bos olabilir).
    """
    chunks = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                          compression="infer", low_memory=False)
    for chunk in reader:
        filtered = chunk[chunk["itemid"].isin(itemids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    result = pd.concat(chunks, ignore_index=True)
    for col in parse_dates:
        result[col] = pd.to_datetime(result[col])
    return result


def _expand_infusions_hourly(inputevents: pd.DataFrame, itemid: int, drug_name: str) -> pd.Series:
    """(starttime, endtime, rate) araliklarindan olusan infuzyon kayitlarini
    saatlik bir doz serisine genisletir.

    Bir saatte birden fazla infuzyon araligi cakisirsa ortalamalari alinir.
    `rate` degeri bos ise (bazi MIMIC-IV bolus kayitlarinda oldugu gibi),
    `amount / sure(saat)` ile efektif bir saatlik doz turetilir.

    Args:
        inputevents: Tek bir itemid/stay_id icin filtrelenmemis inputevents satirlari.
        itemid: Genisletilecek ilacin itemid degeri.
        drug_name: Cikti serisinin adi.

    Returns:
        DatetimeIndex'li (saatlik), drug_name adinda pandas.Series.
    """
    rows = inputevents[inputevents["itemid"] == itemid]
    if rows.empty:
        return pd.Series(dtype=float, name=drug_name)

    hourly_sum: dict[pd.Timestamp, float] = {}
    hourly_count: dict[pd.Timestamp, int] = {}

    for _, row in rows.iterrows():
        start = row["starttime"]
        end = row["endtime"] if pd.notna(row["endtime"]) else start

        rate = row.get("rate")
        if pd.isna(rate):
            amount = row.get("amount")
            duration_h = max((end - start).total_seconds() / 3600.0, 1e-6)
            rate = float(amount) / duration_h if pd.notna(amount) else np.nan
        if pd.isna(rate):
            continue

        start_hour, end_hour = start.floor("h"), end.floor("h")
        if end_hour < start_hour:
            end_hour = start_hour

        for hour in pd.date_range(start_hour, end_hour, freq="h"):
            hourly_sum[hour] = hourly_sum.get(hour, 0.0) + float(rate)
            hourly_count[hour] = hourly_count.get(hour, 0) + 1

    hours = sorted(hourly_sum)
    values = [hourly_sum[h] / hourly_count[h] for h in hours]
    return pd.Series(values, index=pd.DatetimeIndex(hours), name=drug_name)


def _hourly_mean_by_item(df: pd.DataFrame, time_col: str, itemid_map: dict[str, int]) -> pd.DataFrame:
    """chartevents/labevents satirlarini itemid basina saatlik ortalamaya indirger.

    Args:
        df: itemid, valuenum ve time_col sutunlarini iceren DataFrame.
        time_col: Zaman damgasi sutunu ("charttime").
        itemid_map: {sutun_adi: itemid} eslemesi.

    Returns:
        Sutunlari itemid_map anahtarlari olan, saatlik DatetimeIndex'li DataFrame.
    """
    series = {}
    for name, itemid in itemid_map.items():
        sub = df[df["itemid"] == itemid][[time_col, "valuenum"]].dropna()
        if sub.empty:
            series[name] = pd.Series(dtype=float)
            continue
        series[name] = sub.set_index(time_col)["valuenum"].resample("h").mean()
    return pd.DataFrame(series)


def _resolve_stay_window(data_dir: str, stay_id: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """icustays.csv'den verilen stay_id'nin gercek giris/cikis zamanlarini okur.

    labevents.csv, stay_id sutunu icermez (yalnizca subject_id); bir hastanin
    yillar suren tum lab gecmisini tek bir ICU yatisiyla karistirmamak icin
    lablari bu gercek yatis penceresine gore sinirlamak gerekir. icustays.csv
    bulunamazsa cagiran taraf ilac/vital verisinden turetilen bir pencereye
    geri duser.

    Args:
        data_dir: MIMIC-IV kok dizini.
        stay_id: Pencere zamanlari aranacak ICU yatisi.

    Returns:
        (intime, outtime) demeti, ya da icustays.csv/stay_id bulunamazsa None.
    """
    try:
        path = _find_mimic_file(data_dir, "icustays")
    except FileNotFoundError:
        return None

    icustays = pd.read_csv(path, usecols=["stay_id", "intime", "outtime"], compression="infer")
    row = icustays[icustays["stay_id"] == stay_id]
    if row.empty:
        return None
    return pd.to_datetime(row["intime"].iloc[0]), pd.to_datetime(row["outtime"].iloc[0])


def list_mimic_stays(data_dir: str = "mimic_data") -> pd.DataFrame:
    """icustays.csv'den mevcut ICU yatislarinin (stay_id) listesini,
    her birinin inputevents.csv'de hedef ilaclardan (DRUGS) kacina ait
    kaydi oldugu bilgisiyle birlikte dondurur.

    n_drugs bilgisi eklenmistir cunku MIMIC-IV demo veri setinde bircok
    yatisin hedef ilaclarin (norepinefrin/propofol/insulin/heparin) hicbirine
    ait kaydi yoktur; bu bilgi olmadan kullaniciya sunulan stay_id listesinin
    cogu load_mimic_data() cagrildiginda ValueError ile sonuclanir. Liste
    n_drugs'a gore azalan sirada dondurulur, boylece en cok veri iceren
    yatislar en ustte gorunur.

    Args:
        data_dir: MIMIC-IV kok dizini.

    Returns:
        Sutunlar: stay_id, subject_id, intime, outtime, n_drugs.
        n_drugs'a gore azalan sirali pandas.DataFrame.

    Raises:
        FileNotFoundError: icustays.csv data_dir altinda bulunamazsa.
    """
    icustays_path = _find_mimic_file(data_dir, "icustays")
    icustays = pd.read_csv(
        icustays_path, usecols=["stay_id", "subject_id", "intime", "outtime"], compression="infer"
    )
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    try:
        inputevents_path = _find_mimic_file(data_dir, "inputevents")
        inputevents = _read_filtered_csv(
            inputevents_path,
            itemids=list(_MIMIC_DRUG_ITEMIDS.values()),
            usecols=["stay_id", "itemid"],
            parse_dates=[],
        )
        drug_counts = inputevents.groupby("stay_id")["itemid"].nunique().rename("n_drugs")
        icustays = icustays.merge(drug_counts, on="stay_id", how="left")
    except FileNotFoundError:
        icustays["n_drugs"] = np.nan

    icustays["n_drugs"] = icustays["n_drugs"].fillna(0).astype(int)
    return icustays.sort_values("n_drugs", ascending=False).reset_index(drop=True)


def load_mimic_data(data_dir: str = "mimic_data", stay_id: int | None = None,
                     subject_id: int | None = None) -> pd.DataFrame:
    """MIMIC-IV inputevents/chartevents/labevents dosyalarindan tek bir ICU
    yatisina (stay) ait saatlik ilac dozu + vital + lab verisi olusturur.

    Beklenen dosyalar (`data_dir` altinda, duz ya da `icu/`/`hosp/` gibi
    alt klasorlerde; `.csv` veya `.csv.gz` uzantili):
        - inputevents: subject_id, stay_id, starttime, endtime, itemid, rate, amount, ...
        - chartevents: subject_id, stay_id, charttime, itemid, valuenum, ...
        - labevents:   subject_id, charttime, itemid, valuenum, ...
        - icustays:    stay_id, intime, outtime (opsiyonel ama onerilir;
          labevents'i gercek yatis penceresine sinirlamak icin kullanilir)

    Itemid eslemeleri:
        Ilaclar : norepinefrin=221906, propofol=222168, insulin=223257, heparin=225975
        Vitaller: MAP=220052, HR=220045, SpO2=220277
        Lablar  : laktat=50813, glukoz=50931

    Isleyis:
        1. inputevents.csv'den hedef ilaclarin (starttime, endtime, rate)
           araliklari saatlik doz serilerine genisletilir.
        2. Bir stay_id/subject_id verilmemisse, hedef ilaclara ait en cok
           kayda sahip stay_id otomatik secilir (chunked okuma nedeniyle
           filtre once itemid'e, sonra stay_id'ye uygulanir).
        3. chartevents.csv'den MAP/HR/SpO2, labevents.csv'den laktat/glukoz
           saatlik ortalamaya indirgenir.
        4. Tum seriler ortak saatlik bir zaman izgarasina (ilk-son gozlem
           araligi) hizalanir ve fill_missing() ile eksikler doldurulur.

    Args:
        data_dir: CSV dosyalarinin bulundugu dizin.
        stay_id: Analiz edilecek ICU yatisi. None ise otomatik secilir.
        subject_id: stay_id verilmediginde, bu hastaya ait kayitlar arasindan secim yapilir.

    Returns:
        simulate_patient_data() ile ayni formatta (DRUGS + VITALS + saatlik
        DatetimeIndex), ek olarak "lactate" sutunu iceren pandas.DataFrame.

    Raises:
        FileNotFoundError: Beklenen dosyalardan biri data_dir altinda
            (alt klasorler dahil) `.csv` ya da `.csv.gz` olarak bulunamazsa.
        ValueError: Secilen hasta/yatis icin ilac verisi bulunamazsa.
    """
    paths = {name: _find_mimic_file(data_dir, name) for name in ("inputevents", "chartevents", "labevents")}

    # 1) Ilac dozlari
    inputevents = _read_filtered_csv(
        paths["inputevents"],
        itemids=list(_MIMIC_DRUG_ITEMIDS.values()),
        usecols=["subject_id", "stay_id", "starttime", "endtime", "itemid", "rate", "amount"],
        parse_dates=["starttime", "endtime"],
    )
    if inputevents.empty:
        raise ValueError("inputevents.csv icinde hedef ilaclara (itemid) ait kayit bulunamadi.")

    if stay_id is None:
        candidates = inputevents if subject_id is None else inputevents[inputevents["subject_id"] == subject_id]
        if candidates.empty:
            raise ValueError(f"subject_id={subject_id} icin ilac kaydi bulunamadi.")
        stay_id = candidates["stay_id"].value_counts().idxmax()

    inputevents = inputevents[inputevents["stay_id"] == stay_id]
    if inputevents.empty:
        raise ValueError(f"stay_id={stay_id} icin ilac kaydi bulunamadi.")
    resolved_subject_id = inputevents["subject_id"].iloc[0]

    drugs_df = pd.DataFrame({
        name: _expand_infusions_hourly(inputevents, itemid, name)
        for name, itemid in _MIMIC_DRUG_ITEMIDS.items()
    })

    # 2) Vitaller (ayni stay_id ile filtrelenir)
    chartevents = _read_filtered_csv(
        paths["chartevents"],
        itemids=list(_MIMIC_VITAL_ITEMIDS.values()),
        usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    chartevents = chartevents[chartevents["stay_id"] == stay_id]
    vitals_df = _hourly_mean_by_item(chartevents, "charttime", _MIMIC_VITAL_ITEMIDS)

    # 3) Lablar (labevents.csv genelde stay_id icermez -> subject_id ile filtrelenir)
    #    Bir hastanin FARKLI yatislara ait lab kayitlarinin bu yatisa sizmasini
    #    onlemek icin, gercek yatis penceresi (icustays.csv) ile sinirlandirilir.
    labevents = _read_filtered_csv(
        paths["labevents"],
        itemids=list(_MIMIC_LAB_ITEMIDS.values()),
        usecols=["subject_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    labevents = labevents[labevents["subject_id"] == resolved_subject_id]

    stay_window = _resolve_stay_window(data_dir, stay_id)
    if stay_window is None:
        # icustays.csv yoksa, ilac verisinin kapsadigi araligi pencere olarak kullan
        ref_times = [ts for ts in (drugs_df.index.min(), drugs_df.index.max()) if pd.notna(ts)]
        if ref_times:
            stay_window = (min(ref_times), max(ref_times))

    if stay_window is not None:
        buffer = pd.Timedelta(hours=24)
        labevents = labevents[
            (labevents["charttime"] >= stay_window[0] - buffer) &
            (labevents["charttime"] <= stay_window[1] + buffer)
        ]

    labs_df = _hourly_mean_by_item(labevents, "charttime", _MIMIC_LAB_ITEMIDS)

    # 4) Saatlik birlestirme + eksik doldurma
    combined = pd.concat([drugs_df, vitals_df, labs_df], axis=1)
    if combined.empty or combined.dropna(how="all").empty:
        raise ValueError(f"stay_id={stay_id} icin birlestirilebilir veri bulunamadi.")

    if stay_window is not None:
        full_index = pd.date_range(stay_window[0].floor("h"), stay_window[1].ceil("h"), freq="h")
    else:
        full_index = pd.date_range(combined.index.min(), combined.index.max(), freq="h")
    combined = combined.reindex(full_index)
    combined.index.name = "timestamp"

    ordered_cols = DRUGS + VITALS + ["lactate"]
    for col in ordered_cols:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[ordered_cols]

    return fill_missing(combined, method="interpolate")


if __name__ == "__main__":
    print("=== data_manager.py bagimsiz test ===")

    sim = simulate_patient_data(n_hours=72, seed=42)
    print("\nSimule veri (ilk 5 satir):")
    print(sim.head())
    print("\nIstatistik ozeti:")
    print(sim.describe().T[["mean", "std", "min", "max"]])

    sample_path = "simulated_icu_data.csv"
    save_to_csv(sim, sample_path)
    print(f"\nOrnek veri kaydedildi -> {sample_path}")

    reloaded = load_from_csv(sample_path)
    print("\nCSV'den yeniden yuklendi, sekil:", reloaded.shape)

    with_missing = inject_missing(reloaded, frac=0.08, seed=1)
    print("\nEksik veri sayisi (enjeksiyon sonrasi):")
    print(with_missing.isna().sum())

    filled = fill_missing(with_missing, method="interpolate")
    print("\nDoldurma sonrasi kalan NaN sayisi:", int(filled.isna().sum().sum()))

    log_ret = compute_log_returns(filled, columns=DRUGS)
    print("\nLog-getiri (ilk 5 satir, ilaclar):")
    print(log_ret.head())

    print("\n=== list_mimic_stays() testi ===")
    try:
        stays = list_mimic_stays(data_dir="mimic_data")
        print(f"Toplam yatis sayisi: {len(stays)}, hedef ilaca sahip yatis sayisi: {int((stays['n_drugs'] > 0).sum())}")
        print(stays.head(10))
    except FileNotFoundError as e:
        print(f"Atlaniyor - icustays.csv mimic_data/ icinde henuz yok: {e}")

    print("\n=== load_mimic_data() testi ===")
    try:
        mimic_df = load_mimic_data(data_dir="mimic_data")
        print("MIMIC-IV veri sekli:", mimic_df.shape)
        print(mimic_df.head())
    except FileNotFoundError as e:
        print(f"Atlaniyor - MIMIC-IV CSV dosyalari mimic_data/ icinde henuz yok: {e}")
    except ValueError as e:
        print(f"Atlaniyor - {e}")
