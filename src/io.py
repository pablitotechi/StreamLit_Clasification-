"""
io.py — Carga y lectura del dataset.
BCD-7213 · Universidad LEAD · I Cuatrimestre 2026
"""

import pandas as pd
import streamlit as st


DATA_PATH = "data/Big_Black_Money_Dataset.csv"


@st.cache_data(show_spinner="Cargando dataset…")
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Lee el CSV con separador ';' y normaliza columnas básicas."""
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df
