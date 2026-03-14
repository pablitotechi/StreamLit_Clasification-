"""
classification.py — Benchmarking de modelos de clasificación.
Incluye: probabilidad de corte, K-Fold CV y AUC.
BCD-7213 · Universidad LEAD · I Cuatrimestre 2026
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Catálogo de modelos
# ─────────────────────────────────────────────

def get_classifiers() -> dict:
    return {
        "Logistic Regression":     LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":           DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":           RandomForestClassifier(n_estimators=100, max_depth=8,
                                                           random_state=42, n_jobs=-1),
        "Gradient Boosting":       GradientBoostingClassifier(n_estimators=100,
                                                               learning_rate=0.1, random_state=42),
        "K-Nearest Neighbors":     KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes":             GaussianNB(),
        "SVM (RBF)":               SVC(probability=True, kernel="rbf", random_state=42),
    }


# ─────────────────────────────────────────────
# Balanceo con SMOTE
# ─────────────────────────────────────────────

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                random_state: int = 42):
    """Aplica SMOTE para balancear clases minoritarias."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


# ─────────────────────────────────────────────
# Benchmarking con K-Fold + Umbral de corte
# ─────────────────────────────────────────────

def benchmark_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: list,
    n_splits: int = 5,
    cutoff: float = 0.5,
    use_smote: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Entrena y evalúa los modelos seleccionados usando StratifiedKFold.

    Returns:
        metrics_df  — DataFrame con métricas de cada modelo
        fold_results — dict con y_true / y_prob para curvas ROC
    """
    catalog = get_classifiers()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    records = []
    fold_results = {}

    for name in selected_models:
        if name not in catalog:
            continue
        clf = catalog[name]

        auc_scores, acc_scores, f1_scores = [], [], []
        all_y_true, all_y_prob = [], []

        progress = st.progress(0, text=f"Entrenando {name}…")
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # Balanceo dentro del fold (evita data leakage)
            if use_smote:
                try:
                    X_tr, y_tr = apply_smote(X_tr, y_tr, random_state)
                except Exception:
                    pass   # Si SMOTE falla (e.g. solo 1 clase), continúa sin él

            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_te)[:, 1]

            # Aplicar umbral de corte personalizado
            y_pred = (y_prob >= cutoff).astype(int)

            auc_scores.append(roc_auc_score(y_te, y_prob))
            acc_scores.append(accuracy_score(y_te, y_pred))
            f1_scores.append(f1_score(y_te, y_pred, zero_division=0))

            all_y_true.extend(y_te.tolist())
            all_y_prob.extend(y_prob.tolist())

            progress.progress((fold_idx + 1) / n_splits)

        progress.empty()

        # Métricas sobre todo el out-of-fold
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = (all_y_prob >= cutoff).astype(int)

        records.append({
            "Modelo":      name,
            "AUC":         roc_auc_score(all_y_true, all_y_prob),
            "Accuracy":    accuracy_score(all_y_true, all_y_pred),
            "Precision":   precision_score(all_y_true, all_y_pred, zero_division=0),
            "Recall":      recall_score(all_y_true, all_y_pred, zero_division=0),
            "F1-Score":    f1_score(all_y_true, all_y_pred, zero_division=0),
            "CV AUC (µ)":  np.mean(auc_scores),
            "CV AUC (σ)":  np.std(auc_scores),
        })

        fold_results[name] = {
            "y_true": all_y_true,
            "y_prob": all_y_prob,
            "y_pred": all_y_pred,
        }

    metrics_df = pd.DataFrame(records).sort_values("AUC", ascending=False).reset_index(drop=True)
    return metrics_df, fold_results


# ─────────────────────────────────────────────
# Reporte detallado del mejor modelo
# ─────────────────────────────────────────────

def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """Convierte sklearn classification_report a DataFrame."""
    report = classification_report(y_true, y_pred,
                                   target_names=["Bajo Riesgo", "Alto Riesgo"],
                                   output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(4)
