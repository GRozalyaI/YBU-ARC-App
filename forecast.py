"""
forecast.py
-----------
YBU ilac dozu izleme sistemi icin hibrit zaman serisi tahmini.

Sorumluluklari:
    - ARIMA(2,d,1) modeli (d, ADF/KPSS testine gore otomatik secilir)
    - 2 katmanli LSTM (hidden=64, dropout=0.2, seq_len=12) modeli (PyTorch)
    - Ters-hata agirlikli (inverse-error-weighted) hibrit birlestirme
    - Walk-forward backtest
    - MAE / RMSE / MAPE metrikleri

Not: Ortamda TensorFlow/Keras bulunmadigindan LSTM, PyTorch (torch.nn) ile
uygulanmistir; mimari gereksinimler (2 katman, hidden=64, dropout=0.2,
seq_len=12) korunmustur.

Onemli duzeltme: Onceki surumde ARIMA'nin fark alma derecesi (d) daima 0
sabitlenmisti. stat_analysis.py'nin kendi ADF/KPSS testleri ise ham doz
serilerinin genellikle DURGAN OLMADIGINI gosteriyor (yalnizca log-getiri
serileri tutarli sekilde durgan). d=0 ile durgan olmayan bir seriye ARIMA
uydurmak, modelin sabit bir ortalamaya donmesini varsayar ve gercek bir
surukleniz (drift) yakalayamaz - metodolojik olarak yanlistir. Bu surumde
`select_arima_d()` fonksiyonu, verilen seri uzerinde stationarity_report()
calistirip d'yi otomatik secer (durgansa 0, degilse 1); p=2 ve q=1 sabit
kalir.

Bagimsiz calistirma:
    python forecast.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

from stat_analysis import stationarity_report

ARIMA_P = 2
ARIMA_Q = 1
ARIMA_ORDER = (ARIMA_P, 0, ARIMA_Q)  # geriye donuk uyumluluk icin referans; artik varsayilan degil
SEQ_LEN = 12
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


# --------------------------------------------------------------------------
# Metrikler
# --------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ortalama Mutlak Hata."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kok Ortalama Kare Hata."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Ortalama Mutlak Yuzde Hata (%)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, MAPE degerlerini tek dict icinde dondurur."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


# --------------------------------------------------------------------------
# ARIMA
# --------------------------------------------------------------------------

def select_arima_d(series: np.ndarray | pd.Series, alpha: float = 0.05) -> int:
    """ADF/KPSS testine (stat_analysis.stationarity_report) dayanarak ARIMA
    icin fark alma derecesini (d) otomatik secer.

    Karar kurali:
        - stationarity_report verdict == "stationary" ise d=0
        - aksi halde (non_stationary / trend_stationary / difference_stationary) d=1
        - seri cok kisa oldugundan (< 15 gozlem) ya da test basarisiz
          olduğundan guvenilir bir sonuc alinamazsa, guvenli varsayim olarak
          d=1 dondurulur (fark almak, durgan bir seriyi fazladan fark almanin
          (over-differencing) getirdigi hafif verimsizlikten cok, durgan
          OLMAYAN bir seriyi fark almadan birakmanin yol actigi model
          yanlislarindan kacinmayi onceliklendirir).

    Args:
        series: Fark derecesi belirlenecek seri.
        alpha: ADF/KPSS anlamlilik duzeyi.

    Returns:
        0 veya 1.
    """
    s = series if isinstance(series, pd.Series) else pd.Series(np.asarray(series))
    if len(s) < 15:
        return 1
    try:
        report = stationarity_report(s, alpha=alpha)
    except Exception:
        return 1
    return 0 if report["verdict"] == "stationary" else 1


def fit_arima(series: np.ndarray, order: tuple | None = None,
               p: int = ARIMA_P, q: int = ARIMA_Q):
    """ARIMA(p,d,q) modelini verilen seriye uydurur.

    `order` acikca verilmezse, d select_arima_d() ile ADF/KPSS sonucuna gore
    otomatik secilir; p ve q sabit kalir (varsayilan ARIMA(2,*,1)). Bu,
    onceden sabit ARIMA(2,0,1) kullaniminin durgan olmayan ham doz
    serilerinde yol actigi yanlis varsayimi duzeltir (bkz. modul dokstringi).

    Args:
        series: 1B egitim serisi.
        order: Acikca (p,d,q) verilirse dogrudan kullanilir (testler veya
            bilinen bir siparis icin). None ise d otomatik secilir.
        p: order=None oldugunda kullanilacak AR derecesi.
        q: order=None oldugunda kullanilacak MA derecesi.

    Returns:
        Uydurulmus statsmodels ARIMAResults nesnesi (kullanilan (p,d,q)
        siparisi `fitted.model.order` uzerinden okunabilir).
    """
    if order is None:
        order = (p, select_arima_d(series), q)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        fitted = model.fit()
    return fitted


def forecast_arima(fitted, steps: int) -> np.ndarray:
    """Uydurulmus ARIMA modeli ile `steps` adim ileri tahmin uretir."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = fitted.forecast(steps=steps)
    return np.asarray(forecast)


# --------------------------------------------------------------------------
# LSTM (PyTorch)
# --------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """2 katmanli LSTM tabanli tek adim ileri tahminci.

    Mimari: LSTM(hidden=64, num_layers=2, dropout=0.2) -> Linear(hidden, 1)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # son zaman adiminin gizli durumu
        return self.head(last_step).squeeze(-1)


def _make_sequences(values: np.ndarray, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """1B diziyi (X, y) kayan pencere ciftlerine donusturur."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def train_lstm(series: np.ndarray, seq_len: int = SEQ_LEN, epochs: int = 60,
                lr: float = 1e-2, seed: int = 42) -> tuple[LSTMForecaster, dict, list[float]]:
    """LSTM modelini verilen seri uzerinde egitir.

    Seri, egitim oncesi min-max olceklenir (scaler bilgisi geri dondurulur).
    Her epoch'taki egitim (MSE) kaybi kaydedilir ve ogrenme egrisi olarak
    dondurulur - modelin gercekten ogrenip ogrenmedigini (kaybin azalip
    azalmadigini) gozlemlemek icin kullanilir.

    Args:
        series: 1B egitim serisi.
        seq_len: Girdi pencere uzunlugu.
        epochs: Egitim dongusu sayisi.
        lr: Ogrenme orani (Adam).
        seed: Tekrarlanabilirlik icin tohum.

    Returns:
        (egitilmis model, scaler dict {"min": float, "max": float},
         epoch_losses: her epoch sonundaki MSE kaybini iceren, uzunlugu
         `epochs` olan liste)
    """
    torch.manual_seed(seed)

    series = np.asarray(series, dtype=np.float64)
    s_min, s_max = series.min(), series.max()
    scale = (s_max - s_min) if s_max > s_min else 1.0
    scaled = (series - s_min) / scale

    X, y = _make_sequences(scaled, seq_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))

    scaler = {"min": float(s_min), "max": float(s_max), "scale": float(scale)}
    return model, scaler, epoch_losses


def forecast_lstm(model: LSTMForecaster, scaler: dict, history: np.ndarray,
                   steps: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Egitilmis LSTM ile `steps` adim ileri, ozyinelemeli (recursive) tahmin uretir.

    Args:
        model: Egitilmis LSTMForecaster.
        scaler: train_lstm tarafindan dondurulen {"min", "max", "scale"}.
        history: Tahmin baslangici oncesi son gozlemler (en az seq_len uzunlugunda).
        steps: Tahmin edilecek adim sayisi.
        seq_len: Girdi pencere uzunlugu.

    Returns:
        Orijinal olcekte 1B tahmin dizisi (uzunluk = steps).
    """
    model.eval()
    s_min, scale = scaler["min"], scaler["scale"]
    window = list((np.asarray(history[-seq_len:], dtype=np.float64) - s_min) / scale)

    preds_scaled = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
            next_val = model(x).item()
            preds_scaled.append(next_val)
            window.append(next_val)

    preds = np.array(preds_scaled) * scale + s_min
    return preds


# --------------------------------------------------------------------------
# Hibrit birlestirme
# --------------------------------------------------------------------------

def hybrid_forecast(arima_pred: np.ndarray, lstm_pred: np.ndarray,
                     arima_error: float, lstm_error: float, eps: float = 1e-6) -> np.ndarray:
    """Ters-hata agirlikli hibrit tahmin.

    Agirliklar, her modelin (guncel/gecmis) hata metrigiyle ters orantili
    olarak belirlenir: dusuk hatali model daha yuksek agirlik alir.

        w_arima = (1/e_arima) / (1/e_arima + 1/e_lstm)
        w_lstm  = (1/e_lstm)  / (1/e_arima + 1/e_lstm)

    Args:
        arima_pred: ARIMA tahmin dizisi.
        lstm_pred: LSTM tahmin dizisi (ayni uzunlukta).
        arima_error: ARIMA icin referans hata (orn. backtest RMSE).
        lstm_error: LSTM icin referans hata (orn. backtest RMSE).
        eps: Sifira bolmeyi onlemek icin kucuk sabit.

    Returns:
        Hibrit tahmin dizisi.
    """
    inv_arima = 1.0 / (arima_error + eps)
    inv_lstm = 1.0 / (lstm_error + eps)
    total = inv_arima + inv_lstm
    w_arima = inv_arima / total
    w_lstm = inv_lstm / total
    return w_arima * np.asarray(arima_pred) + w_lstm * np.asarray(lstm_pred)


# --------------------------------------------------------------------------
# Walk-forward backtest
# --------------------------------------------------------------------------

def walk_forward_backtest(series: np.ndarray, initial_window: int = 36,
                           horizon: int = 1, step: int = 6,
                           arima_order: tuple | None = None,
                           seq_len: int = SEQ_LEN, lstm_epochs: int = 30) -> dict:
    """Genisleyen pencereli (expanding window) walk-forward backtest uygular.

    Her adimda:
        1. [0, t) verisiyle ARIMA ve LSTM yeniden egitilir. ARIMA icin d,
           arima_order acikca verilmemisse, o anki egitim penceresinin
           durganlik durumuna gore her adimda YENIDEN secilir (select_arima_d);
           veri buyudukce durganlik degisebilir, bu yuzden tek seferlik bir
           secim yerine adim basina yeniden degerlendirme daha dogrudur.
        2. Ikisi de `horizon` adim ileri tahmin uretir.
        3. Onceki adimdaki hatalar kullanilarak ters-hata agirlikli
           hibrit tahmin hesaplanir (ilk adimda esit agirlik kullanilir).
        4. Gercek deger [t, t+horizon) ile karsilastirilir.

    Args:
        series: Tum zaman serisi (1B numpy array).
        initial_window: Ilk egitim penceresinin uzunlugu.
        horizon: Her adimda tahmin edilecek ileri adim sayisi.
        step: Pencerenin her seferinde kac adim ilerleyecegi.
        arima_order: Acikca (p,d,q) verilirse her adimda sabit kullanilir.
            None ise (varsayilan), her adimda fit_arima() araciligiyla
            o pencerenin durganlik durumuna gore d otomatik secilir.
        seq_len: LSTM girdi pencere uzunlugu.
        lstm_epochs: Her yeniden egitimde LSTM epoch sayisi.

    Returns:
        dict: {
            "y_true": np.ndarray,
            "arima_pred": np.ndarray,
            "lstm_pred": np.ndarray,
            "hybrid_pred": np.ndarray,
            "metrics": {"arima": {...}, "lstm": {...}, "hybrid": {...}},
        }
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)

    y_true_all, arima_all, lstm_all, hybrid_all = [], [], [], []
    prev_arima_err, prev_lstm_err = 1.0, 1.0

    t = initial_window
    while t + horizon <= n:
        train = series[:t]
        actual = series[t:t + horizon]

        arima_fitted = fit_arima(train, order=arima_order)
        arima_pred = forecast_arima(arima_fitted, steps=horizon)

        if len(train) > seq_len + 5:
            lstm_model, scaler, _epoch_losses = train_lstm(train, seq_len=seq_len, epochs=lstm_epochs)
            lstm_pred = forecast_lstm(lstm_model, scaler, train, steps=horizon, seq_len=seq_len)
        else:
            lstm_pred = arima_pred.copy()

        hybrid_pred = hybrid_forecast(arima_pred, lstm_pred, prev_arima_err, prev_lstm_err)

        y_true_all.append(actual)
        arima_all.append(arima_pred)
        lstm_all.append(lstm_pred)
        hybrid_all.append(hybrid_pred)

        prev_arima_err = max(rmse(actual, arima_pred), 1e-3)
        prev_lstm_err = max(rmse(actual, lstm_pred), 1e-3)

        t += step

    y_true = np.concatenate(y_true_all)
    arima_pred_all = np.concatenate(arima_all)
    lstm_pred_all = np.concatenate(lstm_all)
    hybrid_pred_all = np.concatenate(hybrid_all)

    return {
        "y_true": y_true,
        "arima_pred": arima_pred_all,
        "lstm_pred": lstm_pred_all,
        "hybrid_pred": hybrid_pred_all,
        "metrics": {
            "arima": compute_metrics(y_true, arima_pred_all),
            "lstm": compute_metrics(y_true, lstm_pred_all),
            "hybrid": compute_metrics(y_true, hybrid_pred_all),
        },
    }


if __name__ == "__main__":
    from data_manager import simulate_patient_data

    print("=== forecast.py bagimsiz test ===")
    df = simulate_patient_data(n_hours=72, seed=42)
    series = df["norepinephrine"].to_numpy()

    print("\n-- Tek seferlik ARIMA + LSTM tahmini (son 12 saat egitim disi birakilarak) --")
    train, test = series[:-12], series[-12:]

    selected_d = select_arima_d(train)
    print(f"select_arima_d() secimi: d={selected_d} (ARIMA(2,{selected_d},1) kullanilacak)")

    arima_fitted = fit_arima(train)
    print("Kullanilan gercek order:", arima_fitted.model.order)
    arima_pred = forecast_arima(arima_fitted, steps=12)

    lstm_model, scaler, epoch_losses = train_lstm(train, epochs=60)
    print(f"Ogrenme egrisi: ilk kayip={epoch_losses[0]:.6f}, son kayip={epoch_losses[-1]:.6f} "
          f"({len(epoch_losses)} epoch)")
    lstm_pred = forecast_lstm(lstm_model, scaler, train, steps=12)

    arima_err = rmse(test, arima_pred)
    lstm_err = rmse(test, lstm_pred)
    hybrid_pred = hybrid_forecast(arima_pred, lstm_pred, arima_err, lstm_err)

    print("ARIMA  metrikleri:", compute_metrics(test, arima_pred))
    print("LSTM   metrikleri:", compute_metrics(test, lstm_pred))
    print("Hibrit metrikleri:", compute_metrics(test, hybrid_pred))

    print("\n-- Walk-forward backtest (norepinephrine) --")
    result = walk_forward_backtest(series, initial_window=36, horizon=1, step=6, lstm_epochs=25)
    for model_name, m in result["metrics"].items():
        print(f"{model_name:8s} -> MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}  MAPE={m['mape']:.2f}%")
