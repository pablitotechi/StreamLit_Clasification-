"""
regression.py — Benchmarking de modelos de regresión.
Incluye: K-Fold CV y métricas estándar de regresión.
BCD-7213 · Universidad LEAD · I Cuatrimestre 2026
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Catálogo de modelos
# ─────────────────────────────────────────────

def get_regressors() -> dict:
    return {
        "Linear Regression":       LinearRegression(),
        "Ridge Regression":        Ridge(alpha=1.0),
        "Lasso Regression":        Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet":              ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "Decision Tree":           DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest":           RandomForestRegressor(n_estimators=100, max_depth=8,
                                                          random_state=42, n_jobs=-1),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100,
                                                              learning_rate=0.1, random_state=42),
        "K-Nearest Neighbors":     KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        "SVR (RBF)":               SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }


# ─────────────────────────────────────────────
# Benchmarking con K-Fold
# ─────────────────────────────────────────────

def benchmark_regressors(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: list,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Entrena y evalúa modelos de regresión mediante KFold.

    Returns:
        metrics_df  — DataFrame con métricas por modelo
        oof_results — dict con y_true / y_pred out-of-fold
    """
    catalog = get_regressors()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    records = []
    oof_results = {}

    for name in selected_models:
        if name not in catalog:
            continue
        reg = catalog[name]

        all_y_true, all_y_pred = [], []
        rmse_folds = []

        progress = st.progress(0, text=f"Entrenando {name}…")
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_te)

            rmse_folds.append(np.sqrt(mean_squared_error(y_te, y_pred)))
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())

            progress.progress((fold_idx + 1) / n_splits)

        progress.empty()

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        records.append({
            "Modelo":        name,
            "R²":            r2_score(all_y_true, all_y_pred),
            "MAE":           mean_absolute_error(all_y_true, all_y_pred),
            "RMSE":          np.sqrt(mean_squared_error(all_y_true, all_y_pred)),
            "MAPE (%)":      mean_absolute_percentage_error(all_y_true, all_y_pred) * 100,
            "CV RMSE (µ)":   np.mean(rmse_folds),
            "CV RMSE (σ)":   np.std(rmse_folds),
        })

        oof_results[name] = {
            "y_true": all_y_true,
            "y_pred": all_y_pred,
        }

    metrics_df = pd.DataFrame(records).sort_values("R²", ascending=False).reset_index(drop=True)
    return metrics_df, oof_results


# ─────────────────────────────────────────────
# Importancia de variables (solo tree-based)
# ─────────────────────────────────────────────

def get_feature_importances(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray | None:
    """Entrena el modelo completo y extrae importancias si disponibles."""
    catalog = get_regressors()
    if model_name not in catalog:
        return None
    reg = catalog[model_name]
    reg.fit(X, y)
    if hasattr(reg, "feature_importances_"):
        return reg.feature_importances_
    elif hasattr(reg, "coef_"):
        return np.abs(reg.coef_)
    return None
