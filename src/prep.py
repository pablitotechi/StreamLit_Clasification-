"""
prep.py — Limpieza, ingeniería de variables y split train/test.
BCD-7213 · Universidad LEAD · I Cuatrimestre 2026
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ─────────────────────────────────────────────
# 1. Limpieza general
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Preprocesando datos…")
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pasos:
      • Parsear fechas
      • Normalizar Amount (USD): quitar puntos extras → float
      • Convertir booleanos texto → int
      • Eliminar filas con valores nulos críticos
      • Crear variables derivadas útiles
    """
    df = df.copy()

    # --- Fecha ---------------------------------------------------------------
    df["Date of Transaction"] = pd.to_datetime(
        df["Date of Transaction"], format="%m/%d/%y %H:%M", errors="coerce"
    )

    # --- Monto ---------------------------------------------------------------
    # El CSV usa puntos como separador de miles (formato europeo invertido).
    # Conservamos solo dígitos y convertimos a float.
    df["Amount_USD"] = (
        df["Amount (USD)"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Log-transform para escalar magnitudes enormes
    df["Amount_log"] = np.log1p(df["Amount_USD"].clip(lower=0))

    # --- Booleanos -----------------------------------------------------------
    df["Reported_bin"] = df["Reported by Authority"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    ).fillna(0).astype(int)

    # --- Shell companies → int -----------------------------------------------
    df["Shell Companies Involved"] = pd.to_numeric(
        df["Shell Companies Involved"], errors="coerce"
    ).fillna(0).astype(int)

    # --- Risk score → int ----------------------------------------------------
    df["Risk_Score"] = pd.to_numeric(
        df["Money Laundering Risk Score"], errors="coerce"
    ).fillna(df["Money Laundering Risk Score"].median() if df["Money Laundering Risk Score"].dtype != object else 5)

    # --- Variables temporales ------------------------------------------------
    df["Year"]  = df["Date of Transaction"].dt.year
    df["Month"] = df["Date of Transaction"].dt.month
    df["Day"]   = df["Date of Transaction"].dt.day
    df["Hour"]  = df["Date of Transaction"].dt.hour
    df["DayOfWeek"] = df["Date of Transaction"].dt.dayofweek

    # --- Target de clasificación: Alto riesgo (Risk_Score >= 7) --------------
    df["High_Risk"] = (df["Risk_Score"] >= 7).astype(int)

    # --- Eliminar nulos en columnas clave ------------------------------------
    df.dropna(subset=["Date of Transaction", "Amount_USD"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ─────────────────────────────────────────────
# 2. Encoding de categorías
# ─────────────────────────────────────────────

CATEGORICAL_COLS = [
    "Country", "Transaction Type", "Industry",
    "Destination Country", "Source of Money", "Tax Haven Country"
]


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode columnas categóricas. Retorna df codificado + encoders."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


# ─────────────────────────────────────────────
# 3. Feature sets por tarea
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "Amount_log", "Shell Companies Involved", "Reported_bin",
    "Risk_Score", "Year", "Month", "Hour", "DayOfWeek",
    "Country_enc", "Transaction Type_enc", "Industry_enc",
    "Destination Country_enc", "Source of Money_enc", "Tax Haven Country_enc",
]

TARGET_CLASSIFICATION = "High_Risk"
TARGET_REGRESSION     = "Amount_log"


def get_Xy(df: pd.DataFrame, task: str = "classification"):
    """
    Retorna (X, y, feature_names) para clasificación o regresión.
    task: 'classification' | 'regression'
    """
    df_enc, _ = encode_features(df)

    # Asegurar que todas las columnas existen
    available = [c for c in FEATURE_COLS if c in df_enc.columns]

    target = TARGET_CLASSIFICATION if task == "classification" else TARGET_REGRESSION

    # Para regresión, excluir Amount_log del set de features (evitar data leakage)
    if task == "regression":
        feats = [c for c in available if c != "Amount_log"]
    else:
        feats = available

    X = df_enc[feats].copy()
    y = df_enc[target].copy()

    # Escalar
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feats)

    return X_scaled, y, feats


# ─────────────────────────────────────────────
# 4. Preparación para Series de Tiempo
# ─────────────────────────────────────────────

def get_timeseries(df: pd.DataFrame, freq: str = "D") -> pd.Series:
    """
    Agrega el monto (Amount_log) por frecuencia temporal.
    freq: 'D' = diario, 'W' = semanal, 'ME' = mensual
    """
    ts = (
        df.set_index("Date of Transaction")["Amount_log"]
        .resample(freq)
        .mean()
        .dropna()
    )
    return ts
