"""
timeseries.py — Benchmarking de Series de Tiempo.
Métodos: Neural Network, Holt-Winters, HW Calibrado, ARIMA, ARIMA Calibrado.
BCD-7213 · Universidad LEAD · I Cuatrimestre 2026
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# TensorFlow / Keras — carga diferida para no bloquear si no está disponible
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────

def ts_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    return {"Modelo": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


# ─────────────────────────────────────────────
# 1. Holt-Winters
# ─────────────────────────────────────────────

def holtwinters_forecast(train: np.ndarray, h: int,
                         seasonal_periods: int = 7,
                         trend: str = "add",
                         seasonal: str = "add") -> np.ndarray:
    """Holt-Winters con parámetros fijos."""
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal if len(train) >= 2 * seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=False,
                    smoothing_level=0.3,
                    smoothing_trend=0.1,
                    smoothing_seasonal=0.1)
    return fit.forecast(h)


# ─────────────────────────────────────────────
# 2. Holt-Winters Calibrado (optimizado)
# ─────────────────────────────────────────────

def holtwinters_calibrated_forecast(train: np.ndarray, h: int,
                                     seasonal_periods: int = 7) -> np.ndarray:
    """Holt-Winters con optimización automática de parámetros."""
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add" if len(train) >= 2 * seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    return fit.forecast(h)


# ─────────────────────────────────────────────
# 3. ARIMA
# ─────────────────────────────────────────────

def arima_forecast(train: np.ndarray, h: int,
                   order: tuple = (2, 1, 2)) -> np.ndarray:
    """ARIMA con orden fijo (p,d,q)."""
    model = ARIMA(train, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=h)
    return np.array(forecast)


# ─────────────────────────────────────────────
# 4. ARIMA Calibrado (búsqueda de mejor orden)
# ─────────────────────────────────────────────

def _check_stationarity(series: np.ndarray) -> int:
    """Retorna d recomendado basado en test ADF."""
    result = adfuller(series, autolag="AIC")
    return 0 if result[1] < 0.05 else 1


def arima_calibrated_forecast(train: np.ndarray, h: int) -> np.ndarray:
    """Búsqueda de grilla simple sobre p, d, q para minimizar AIC."""
    d = _check_stationarity(train)
    best_aic = np.inf
    best_order = (1, d, 1)

    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(train, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    model = ARIMA(train, order=best_order)
    fit = model.fit()
    return np.array(fit.forecast(steps=h)), best_order


# ─────────────────────────────────────────────
# 5. Red Neuronal (LSTM)
# ─────────────────────────────────────────────

def _create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def neural_network_forecast(train: np.ndarray, h: int,
                             lookback: int = 14,
                             epochs: int = 30,
                             batch_size: int = 16) -> np.ndarray:
    """LSTM univariada con escalado Min-Max."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible. Instálalo con: pip install tensorflow")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()

    X_tr, y_tr = _create_sequences(train_scaled, lookback)
    X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1))

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=[es])

    # Predicción iterativa h pasos
    last_seq = train_scaled[-lookback:].copy()
    preds = []
    for _ in range(h):
        inp = last_seq.reshape(1, lookback, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], pred)

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_inv


# ─────────────────────────────────────────────
# Benchmarking integrado
# ─────────────────────────────────────────────

def benchmark_timeseries(
    ts: pd.Series,
    test_size: float = 0.20,
    seasonal_periods: int = 7,
    selected_methods: list | None = None,
    nn_epochs: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Divide train/test, entrena todos los métodos y compara métricas.

    Returns:
        metrics_df  — DataFrame con MAE, RMSE, MAPE por método
        forecasts   — dict con predicciones de cada método
    """
    n = len(ts)
    split = int(n * (1 - test_size))
    train_vals = ts.iloc[:split].values.astype(float)
    test_vals  = ts.iloc[split:].values.astype(float)
    test_idx   = ts.index[split:]
    h = len(test_vals)

    if selected_methods is None:
        selected_methods = [
            "Holt-Winters",
            "Holt-Winters Calibrado",
            "ARIMA",
            "ARIMA Calibrado",
            "Deep Learning (LSTM)",
        ]

    records = []
    forecasts = {
        "train": (ts.index[:split], train_vals),
        "test":  (test_idx, test_vals),
    }
    extra_info = {}

    for method in selected_methods:
        try:
            if method == "Holt-Winters":
                pred = holtwinters_forecast(train_vals, h, seasonal_periods)

            elif method == "Holt-Winters Calibrado":
                pred = holtwinters_calibrated_forecast(train_vals, h, seasonal_periods)

            elif method == "ARIMA":
                pred = arima_forecast(train_vals, h)

            elif method == "ARIMA Calibrado":
                pred, best_order = arima_calibrated_forecast(train_vals, h)
                extra_info[method] = f"Mejor orden ARIMA: {best_order}"

            elif method == "Deep Learning (LSTM)":
                if not TF_AVAILABLE:
                    records.append({"Modelo": method, "MAE": np.nan,
                                    "RMSE": np.nan, "MAPE (%)": np.nan})
                    continue
                pred = neural_network_forecast(train_vals, h, epochs=nn_epochs)

            else:
                continue

            pred = np.array(pred).flatten()[:h]
            forecasts[method] = (test_idx, pred)
            records.append(ts_metrics(test_vals[:len(pred)], pred, method))

        except Exception as e:
            records.append({"Modelo": method, "MAE": np.nan,
                            "RMSE": np.nan, "MAPE (%)": np.nan,
                            "Error": str(e)})

    metrics_df = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    return metrics_df, forecasts, extra_info
