"""
stat_analysis.py
----------------
YBU ilac dozu izleme sistemi icin durganlik (stationarity) analizi.

Sorumluluklari:
    - ADF (Augmented Dickey-Fuller) testi
    - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) testi
    - Iki testin birlikte yorumlanmasi (durganlik sonucu dict olarak)

ADF ve KPSS'in sifir hipotezleri tersttir:
    - ADF   H0: seri durgan DEGIL (birim kok var)      -> p < 0.05 ise durgan
    - KPSS  H0: seri durgan                              -> p < 0.05 ise durgan DEGIL

Bagimsiz calistirma:
    python stat_analysis.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series: pd.Series, alpha: float = 0.05, autolag: str = "AIC") -> dict:
    """Augmented Dickey-Fuller durganlik testini uygular.

    Args:
        series: Test edilecek zaman serisi (NaN icermemeli).
        alpha: Anlamlilik duzeyi.
        autolag: Gecikme secim yontemi (statsmodels adfuller parametresi).

    Returns:
        dict: {
            "test": "ADF",
            "statistic": float,
            "p_value": float,
            "n_lags": int,
            "n_obs": int,
            "critical_values": dict,
            "is_stationary": bool,   # p_value < alpha
        }
    """
    clean = series.dropna()
    stat, p_value, n_lags, n_obs, crit_values, _ = adfuller(clean, autolag=autolag)
    return {
        "test": "ADF",
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_lags": int(n_lags),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        "is_stationary": bool(p_value < alpha),
        "alpha": alpha,
    }


def kpss_test(series: pd.Series, alpha: float = 0.05, regression: str = "c") -> dict:
    """KPSS durganlik testini uygular.

    Args:
        series: Test edilecek zaman serisi (NaN icermemeli).
        alpha: Anlamlilik duzeyi.
        regression: "c" (sabit etrafinda durganlik) veya
            "ct" (trend etrafinda durganlik).

    Returns:
        dict: {
            "test": "KPSS",
            "statistic": float,
            "p_value": float,
            "n_lags": int,
            "critical_values": dict,
            "is_stationary": bool,  # p_value >= alpha
        }
    """
    clean = series.dropna()
    with warnings.catch_warnings():
        # p-degeri tablo sinirlarinin disinda kaldiginda statsmodels uyari basar;
        # bu durumda p_value zaten interpolasyonla en yakin sinira sabitlenir.
        warnings.simplefilter("ignore")
        stat, p_value, n_lags, crit_values = kpss(clean, regression=regression, nlags="auto")
    return {
        "test": "KPSS",
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_lags": int(n_lags),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        "is_stationary": bool(p_value >= alpha),
        "alpha": alpha,
    }


def stationarity_report(series: pd.Series, alpha: float = 0.05) -> dict:
    """ADF ve KPSS sonuclarini birlestirip nihai bir yorum uretir.

    Olasi verdict degerleri:
        - "stationary": her iki test de durgan diyor
        - "non_stationary": her iki test de durgan degil diyor
        - "trend_stationary": ADF durgan degil, KPSS durgan
          (fark alma yerine trend temizleme dusunulebilir)
        - "difference_stationary": ADF durgan, KPSS durgan degil
          (fark alma onerilir)

    Args:
        series: Test edilecek zaman serisi.
        alpha: Anlamlilik duzeyi.

    Returns:
        dict: {"adf": {...}, "kpss": {...}, "verdict": str}
    """
    adf_res = adf_test(series, alpha=alpha)
    kpss_res = kpss_test(series, alpha=alpha)

    if adf_res["is_stationary"] and kpss_res["is_stationary"]:
        verdict = "stationary"
    elif not adf_res["is_stationary"] and not kpss_res["is_stationary"]:
        verdict = "non_stationary"
    elif not adf_res["is_stationary"] and kpss_res["is_stationary"]:
        verdict = "trend_stationary"
    else:
        verdict = "difference_stationary"

    return {"adf": adf_res, "kpss": kpss_res, "verdict": verdict}


if __name__ == "__main__":
    from data_manager import DRUGS, compute_log_returns, simulate_patient_data

    print("=== stat_analysis.py bagimsiz test ===")
    df = simulate_patient_data(n_hours=72, seed=42)
    log_returns = compute_log_returns(df, columns=DRUGS).dropna()

    for drug in DRUGS:
        print(f"\n--- {drug} (ham doz serisi) ---")
        report = stationarity_report(df[drug])
        print(f"ADF  p={report['adf']['p_value']:.4f}  durgan={report['adf']['is_stationary']}")
        print(f"KPSS p={report['kpss']['p_value']:.4f}  durgan={report['kpss']['is_stationary']}")
        print(f"Sonuc: {report['verdict']}")

        print(f"--- {drug} (log-getiri serisi) ---")
        report_lr = stationarity_report(log_returns[f"{drug}_log_return"])
        print(f"ADF  p={report_lr['adf']['p_value']:.4f}  durgan={report_lr['adf']['is_stationary']}")
        print(f"KPSS p={report_lr['kpss']['p_value']:.4f}  durgan={report_lr['kpss']['is_stationary']}")
        print(f"Sonuc: {report_lr['verdict']}")
