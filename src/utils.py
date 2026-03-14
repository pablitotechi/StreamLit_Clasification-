"""
utils.py — Visualizaciones compartidas y métricas auxiliares.
BCD-7213 · Universidad LEAD · I Cuatrimestre 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay
)
import streamlit as st

# ── Paleta institucional ───────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1a3a5c",
    "secondary": "#e8601c",
    "accent":    "#2ecc71",
    "neutral":   "#95a5a6",
    "bg":        "#f0f4f8",
}

sns.set_theme(style="whitegrid", palette="deep")


# ─────────────────────────────────────────────
# ROC / AUC
# ─────────────────────────────────────────────

def plot_roc_curves(results: dict) -> go.Figure:
    """
    results: {model_name: {"y_true": ..., "y_prob": ...}}
    Devuelve figura Plotly con todas las curvas ROC superpuestas.
    """
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="grey", width=1))

    palette = px.colors.qualitative.Set2
    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC={roc_auc:.4f})",
            line=dict(color=palette[i % len(palette)], width=2.5)
        ))

    fig.update_layout(
        title="Curvas ROC – Comparación de Modelos",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        legend=dict(x=0.55, y=0.05),
        height=480, template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────
# Matriz de confusión
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bajo Riesgo", "Alto Riesgo"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Benchmark table
# ─────────────────────────────────────────────

def styled_benchmark_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el dataframe con formato para st.dataframe."""
    return df_metrics.style.background_gradient(
        cmap="YlGn", subset=[c for c in df_metrics.columns if c != "Modelo"]
    ).format("{:.4f}", subset=[c for c in df_metrics.columns if c != "Modelo"])


# ─────────────────────────────────────────────
# Gráfico de importancia de variables
# ─────────────────────────────────────────────

def plot_feature_importance(importances: np.ndarray,
                            feature_names: list,
                            title: str = "Importancia de Variables") -> go.Figure:
    idx = np.argsort(importances)[::-1][:15]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker_color=COLORS["secondary"],
    ))
    fig.update_layout(
        title=title, height=420,
        xaxis_title="Importancia",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────
# Curva Precision–Recall por umbral de corte
# ─────────────────────────────────────────────

def plot_cutoff_analysis(y_true, y_prob,
                         title: str = "Análisis de Probabilidad de Corte") -> go.Figure:
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        precisions.append(p); recalls.append(r); f1s.append(f)

    best_t = thresholds[np.argmax(f1s)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, name="Precision", line=dict(color="#1a3a5c")))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls,    name="Recall",    line=dict(color="#e8601c")))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s,        name="F1-Score",  line=dict(color="#2ecc71")))
    fig.add_vline(x=best_t, line_dash="dash", line_color="purple",
                  annotation_text=f"Mejor umbral: {best_t:.2f}", annotation_position="top right")

    fig.update_layout(
        title=title, xaxis_title="Umbral de Corte",
        yaxis_title="Métrica", height=420, template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig, best_t


# ─────────────────────────────────────────────
# KPI cards helper
# ─────────────────────────────────────────────

def kpi_row(metrics: dict):
    """Muestra métricas clave en columnas tipo KPI."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label=label, value=f"{value:.4f}" if isinstance(value, float) else value)


# ─────────────────────────────────────────────
# Predicciones vs Reales (regresión / series)
# ─────────────────────────────────────────────

def plot_pred_vs_real(y_true, y_pred,
                      index=None,
                      title: str = "Real vs Predicho",
                      is_ts: bool = False) -> go.Figure:
    x = index if index is not None else np.arange(len(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.array(y_true).flatten(),
                             name="Real", line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=np.array(y_pred).flatten(),
                             name="Predicho", line=dict(color=COLORS["secondary"],
                                                         width=2, dash="dot")))
    fig.update_layout(
        title=title, height=420, template="plotly_white",
        xaxis_title="Fecha" if is_ts else "Observación",
        yaxis_title="Amount (log)" if is_ts else "Valor",
    )
    return fig
