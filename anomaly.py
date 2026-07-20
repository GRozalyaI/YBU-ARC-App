"""
anomaly.py
----------
YBU ilac dozu izleme sistemi icin anomali tespiti.

Sorumluluklari:
    - Kayan (rolling) Z-Score tespiti (pencere=24, esik=2.5)
    - Isolation Forest tespiti
    - LSTM Autoencoder (yeniden yapilandirma hatasi tabanli) tespiti
    - Uc yontemin oy birligi (ensemble) ile birlestirilmesi
    - F1 / Precision / Recall hesaplama

Bagimsiz calistirma:
    python anomaly.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

ROLLING_WINDOW = 24
ROLLING_THRESHOLD = 2.5
AE_SEQ_LEN = 12


# --------------------------------------------------------------------------
# 1) Kayan Z-Score
# --------------------------------------------------------------------------

def rolling_zscore_values(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Kayan ortalama/std'ye gore ham Z-Score degerlerini hesaplar (esik uygulanmaz).

    z_t = (x_t - rolling_mean_t) / rolling_std_t

    Args:
        series: Zaman serisi.
        window: Kayan pencere uzunlugu (saat).

    Returns:
        Ayni index'e sahip float pandas.Series (rolling_std=0 veya yetersiz
        pencere durumunda NaN).
    """
    rolling_mean = series.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def rolling_zscore_detect(series: pd.Series, window: int = ROLLING_WINDOW,
                           threshold: float = ROLLING_THRESHOLD) -> pd.Series:
    """Kayan ortalama/std'ye gore Z-Score anomali tespiti.

    |z_t| > threshold ise anomali.

    Args:
        series: Zaman serisi.
        window: Kayan pencere uzunlugu (saat).
        threshold: Anomali esigi (|z| > threshold).

    Returns:
        Ayni index'e sahip boolean pandas.Series (True = anomali).
    """
    z = rolling_zscore_values(series, window=window)
    anomalies = z.abs() > threshold
    return anomalies.fillna(False)


# --------------------------------------------------------------------------
# 2) Isolation Forest
# --------------------------------------------------------------------------

# Her ilacin, data_manager.simulate_patient_data() dokstringinde belgelenen
# fizyolojik korelasyonuna gore eslendigi vital bulgu(lar). Isolation Forest'i
# yalnizca tek bir ilac sutunuyla (tek degiskenli) cagirmak, algoritmanin asil
# gucunu (birden fazla degiskenin BIRLIKTE olagandisi olup olmadigini tespit
# etme) kullanmadan onu Z-Score'un kaba bir kopyasina indirger. Bu esleme,
# ilac dozunu + onunla fizyolojik olarak baglantili vital(leri) birlikte
# vererek gercek cok degiskenli tespit yapilmasini saglar.
DRUG_VITAL_LINKS = {
    "norepinephrine": ["map"],
    "propofol": ["heart_rate"],
    "insulin": ["glucose"],
    "heparin": ["heart_rate"],
}


def isolation_forest_features_for_drug(drug: str, available_columns: list[str]) -> list[str]:
    """Bir ilac icin Isolation Forest'e verilecek cok degiskenli ozellik
    listesini olusturur: ilacin kendisi + DRUG_VITAL_LINKS'te belirtilen,
    df'de gercekten mevcut olan iliskili vital bulgu(lar).

    Args:
        drug: Ilac sutunu adi.
        available_columns: Elde mevcut olan DataFrame sutunlari (orn. df.columns).

    Returns:
        [drug, ilgili_vital1, ...] seklinde, yalnizca available_columns
        icinde bulunan sutunlardan olusan liste (en az [drug] icerir).
    """
    related = [v for v in DRUG_VITAL_LINKS.get(drug, []) if v in available_columns]
    return [drug, *related]


def isolation_forest_detect(df: pd.DataFrame, features: list[str] | None = None,
                             contamination: float = 0.05, seed: int = 42) -> pd.Series:
    """Isolation Forest ile cok degiskenli anomali tespiti.

    Args:
        df: Ozellik sutunlarini iceren DataFrame.
        features: Kullanilacak sutunlar; None ise tum sutunlar. Gercekten
            cok degiskenli (multivariate) bir tespit icin en az 2 sutun
            verilmesi onerilir (bkz. isolation_forest_features_for_drug);
            tek sutunla cagirmak algoritmayi fiilen tek degiskenli hale
            getirir ve Rolling Z-Score ile buyuk olcude ortusmesine yol acar.
        contamination: Beklenen anomali orani.
        seed: Rastgelelik tohumu.

    Returns:
        df ile ayni index'e sahip boolean pandas.Series (True = anomali).
    """
    cols = features or list(df.columns)
    X = df[cols].to_numpy()
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    labels = model.fit_predict(X)  # -1 anomali, 1 normal
    return pd.Series(labels == -1, index=df.index)


# --------------------------------------------------------------------------
# 3) LSTM Autoencoder
# --------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """Kayan pencere yeniden yapilandirmasi ile anomali tespiti icin
    basit bir LSTM Autoencoder (encoder-decoder)."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, seq_len: int = AE_SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]  # (batch, hidden_size) - sikistirilmis temsil
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)  # (batch, seq_len, input_size)


def _windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    """1B diziyi ortusen (overlapping) pencerelere ayirir."""
    return np.array([values[i:i + seq_len] for i in range(len(values) - seq_len + 1)])


def lstm_autoencoder_detect(series: pd.Series, seq_len: int = AE_SEQ_LEN,
                             epochs: int = 100, threshold_std: float = 3.0,
                             seed: int = 42) -> pd.Series:
    """LSTM Autoencoder yeniden yapilandirma hatasi ile anomali tespiti.

    Model normal davranisi ogrenmek uzere tum seri uzerinde egitilir
    (etiketsiz/unsupervised); pencere basina yeniden yapilandirma hatasi
    ortalama + threshold_std * std degerini astiginda o pencerenin son
    zaman noktasi anomali olarak isaretlenir.

    Args:
        series: Zaman serisi.
        seq_len: Pencere uzunlugu.
        epochs: Egitim dongusu sayisi.
        threshold_std: Anomali esigi (ortalama hatanin kac std uzeri).
        seed: Rastgelelik tohumu.

    Returns:
        series ile ayni index'e sahip boolean pandas.Series (True = anomali).
        Ilk (seq_len - 1) nokta icin yeterli pencere olmadigindan False atanir.
    """
    torch.manual_seed(seed)

    values = series.to_numpy(dtype=np.float64)
    v_min, v_max = values.min(), values.max()
    scale = (v_max - v_min) if v_max > v_min else 1.0
    scaled = (values - v_min) / scale

    windows = _windows(scaled, seq_len)  # (N, seq_len)
    X = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)

    model = LSTMAutoencoder(input_size=1, hidden_size=32, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(X).numpy()
    sq_error = (windows[..., None] - recon) ** 2  # (N, seq_len, 1)
    sq_error = sq_error.squeeze(-1)  # (N, seq_len)

    # Her global zaman noktasi, kendisini iceren tum penceresel hata
    # tahminlerinin ortalamasi ile puanlanir (tek bir pencerenin son
    # noktasina indirgemek, pencerenin ortasindaki nokta anomalilerini
    # sonraki normal noktaya kaydirir).
    n = len(values)
    n_windows = windows.shape[0]
    error_sum = np.zeros(n)
    error_count = np.zeros(n)
    for i in range(n_windows):
        error_sum[i:i + seq_len] += sq_error[i]
        error_count[i:i + seq_len] += 1
    per_point_error = error_sum / np.maximum(error_count, 1)

    mean_err, std_err = per_point_error.mean(), per_point_error.std()
    anomalies = per_point_error > (mean_err + threshold_std * std_err)
    return pd.Series(anomalies, index=series.index)


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

def ensemble_detect(masks: dict[str, pd.Series], method: str = "majority") -> pd.Series:
    """Birden fazla anomali maskesini birlestirir.

    Args:
        masks: {"yontem_adi": boolean pd.Series} seklinde sozluk.
        method: "majority" (coğunluk oyu) veya "union" (herhangi biri True ise True).

    Returns:
        Birlesik boolean pandas.Series.
    """
    df_masks = pd.DataFrame(masks).astype(int)
    if method == "union":
        return df_masks.any(axis=1)
    if method == "majority":
        votes_needed = (df_masks.shape[1] // 2) + 1
        return df_masks.sum(axis=1) >= votes_needed
    raise ValueError(f"Bilinmeyen birlestirme yontemi: {method}")


# --------------------------------------------------------------------------
# Metrikler
# --------------------------------------------------------------------------

def classification_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict:
    """Precision / Recall / F1 hesaplar (pozitif sinif = anomali/True).

    Args:
        y_true: Gercek anomali etiketleri (bool).
        y_pred: Tahmin edilen anomali etiketleri (bool).

    Returns:
        dict: {"precision": float, "recall": float, "f1": float}
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    return {
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }


def _inject_anomalies(df: pd.DataFrame, column: str, n_anomalies: int = 5,
                       magnitude: float = 6.0, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """Test amacli, bilinen anomali noktalari enjekte eder (gercek etiket uretimi icin)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    # baslangic/bitisten uzak, birbirine cok yakin olmayan indeksler sec
    candidate_idx = np.arange(30, n - 5)
    idx = rng.choice(candidate_idx, size=n_anomalies, replace=False)
    std = out[column].std()
    for i in idx:
        out.iloc[i, out.columns.get_loc(column)] += rng.choice([-1, 1]) * magnitude * std
    true_labels = np.zeros(n, dtype=bool)
    true_labels[idx] = True
    return out, true_labels


if __name__ == "__main__":
    from data_manager import simulate_patient_data

    print("=== anomaly.py bagimsiz test ===")
    df = simulate_patient_data(n_hours=72, seed=42)
    df_anom, true_labels = _inject_anomalies(df, column="norepinephrine", n_anomalies=6, magnitude=5.0)

    series = df_anom["norepinephrine"]

    if_features = isolation_forest_features_for_drug("norepinephrine", df_anom.columns.tolist())
    print(f"Isolation Forest ozellikleri (cok degiskenli): {if_features}")

    z_mask = rolling_zscore_detect(series)
    if_mask = isolation_forest_detect(df_anom, features=if_features, contamination=6 / len(series))
    ae_mask = lstm_autoencoder_detect(series, epochs=120)

    ens_majority = ensemble_detect(
        {"zscore": z_mask, "isolation_forest": if_mask, "autoencoder": ae_mask}, method="majority"
    )
    ens_union = ensemble_detect(
        {"zscore": z_mask, "isolation_forest": if_mask, "autoencoder": ae_mask}, method="union"
    )

    print(f"\nEnjekte edilen gercek anomali sayisi: {true_labels.sum()}")
    for name, mask in [
        ("Rolling Z-Score", z_mask),
        ("Isolation Forest", if_mask),
        ("LSTM Autoencoder", ae_mask),
        ("Ensemble (majority)", ens_majority),
        ("Ensemble (union)", ens_union),
    ]:
        m = classification_metrics(true_labels, mask.to_numpy())
        print(f"{name:22s} -> Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}"
              f"  (tespit sayisi={int(mask.sum())})")
