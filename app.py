"""
app.py — Aplicación principal Streamlit.
BCD-7213 · Minería de Datos Avanzada · Universidad LEAD
I Cuatrimestre 2026 · Dr. Juan Murillo-Morera
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Módulos del proyecto ───────────────────────────────────────────────────────
from src.io import load_dataset
from src.prep import clean_dataset, get_Xy, get_timeseries
from src.classification import (
    get_classifiers, benchmark_classifiers, classification_report_df
)
from src.regression import (
    get_regressors, benchmark_regressors, get_feature_importances
)
from src.timeseries import benchmark_timeseries
from src.utils import (
    plot_roc_curves, plot_confusion_matrix, styled_benchmark_table,
    plot_feature_importance, plot_cutoff_analysis, kpi_row,
    plot_pred_vs_real, COLORS
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuración de página
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Caso de Estudio 1 · Minería de Datos Avanzada",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .main-title {
        background: linear-gradient(135deg, #1a3a5c 0%, #2c6fad 100%);
        color: white; padding: 1.4rem 2rem; border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(26,58,92,0.3);
    }
    .main-title h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .main-title p  { margin: 0.2rem 0 0; font-size: 0.85rem; opacity: 0.85; }

    /* Section badges */
    .section-badge {
        display: inline-block;
        background: #e8601c; color: white;
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; margin-bottom: 0.6rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f0f4f8; border-radius: 10px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #1a3a5c;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a3a5c; }
    [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #bdc3c7 !important; }

    /* Dividers */
    hr { border-color: #d5dde8; margin: 1.2rem 0; }

    /* Info boxes */
    .info-box {
        background: #eaf4fb; border-left: 4px solid #2c6fad;
        padding: 0.8rem 1.1rem; border-radius: 6px; margin: 0.8rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Header global
st.markdown("""
<div class="main-title">
    <h1> Minería de Datos Avanzada — Caso de Estudio #1</h1>
    <p>Universidad LEAD · I Cuatrimestre 2026 · Dr. Juan Murillo-Morera &nbsp;|&nbsp;
       Dataset: <b>Big Black Money Dataset</b> (9,999 transacciones financieras)</p>
</div>
""", unsafe_allow_html=True)


# Sidebar — Navegación y parámetros globales
with st.sidebar:
    st.image("img/PPPZILC6CNEV3MNGCPLFQSQJVI-301940457.png",
             width=140)
    st.markdown("---")
    st.markdown("### Módulo")
    module = st.radio(
        "",
        ["Exploración de Datos",
         "Clasificación",
         "Regresión",
         "Series de Tiempo"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Parámetros Globales")
    n_splits   = st.slider("K-Fold (splits)", 3, 10, 5)
    test_size  = st.slider("Tamaño Test (%)", 10, 40, 20) / 100
    random_seed = st.number_input("Semilla aleatoria", 0, 9999, 42)
    st.markdown("---")
    st.markdown("<small style='color:#95a5a6'>BCD-7213 · LEAD 2026</small>",
                unsafe_allow_html=True)


# Carga y preprocesamiento
raw_df = load_dataset()
df     = clean_dataset(raw_df)


# MÓDULO 1 — Exploración de Datos
if module == "Exploración de Datos":
    st.markdown('<span class="section-badge">Exploración de Datos</span>',
                unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transacciones", f"{len(df):,}")
    col2.metric("Variables", f"{df.shape[1]}")
    col3.metric("% Alto Riesgo", f"{df['High_Risk'].mean()*100:.1f}%")
    col4.metric("Período", f"{df['Year'].min()} – {df['Year'].max()}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Muestra de Datos", "Distribuciones", "Geografía", "Correlaciones"
    ])

    # ── Tab 1: Muestra ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Vista previa del dataset limpio")
        st.dataframe(df.head(200), use_container_width=True, height=380)
        with st.expander("Descripción estadística"):
            st.dataframe(df.describe().T.style.format("{:.2f}"),
                         use_container_width=True)

    # ── Tab 2: Distribuciones ───────────────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Risk_Score", nbins=10,
                               color_discrete_sequence=[COLORS["primary"]],
                               title="Distribución del Risk Score")
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(df, x="Amount_log", nbins=40,
                               color_discrete_sequence=[COLORS["secondary"]],
                               title="Distribución del Monto (log)")
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            counts = df["Transaction Type"].value_counts().reset_index()
            fig = px.bar(counts, x="Transaction Type", y="count",
                         color="Transaction Type",
                         title="Tipos de Transacción",
                         template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = px.pie(df, names="High_Risk",
                         title="Balance de Clases (Alto Riesgo)",
                         color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]])
            fig.update_traces(labels=["Bajo Riesgo", "Alto Riesgo"])
            fig.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Geografía ────────────────────────────────────────────────────
    with tab3:
        country_risk = df.groupby("Country")["High_Risk"].mean().reset_index()
        country_risk.columns = ["País", "% Alto Riesgo"]
        country_risk["% Alto Riesgo"] *= 100

        fig = px.bar(
            country_risk.sort_values("% Alto Riesgo", ascending=False).head(20),
            x="País", y="% Alto Riesgo",
            color="% Alto Riesgo", color_continuous_scale="Reds",
            title="% de Transacciones Alto Riesgo por País de Origen",
            template="plotly_white", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        dst = df.groupby("Destination Country")["Amount_log"].mean().reset_index()
        dst.columns = ["País Destino", "Monto Promedio (log)"]
        fig2 = px.bar(
            dst.sort_values("Monto Promedio (log)", ascending=False).head(15),
            x="País Destino", y="Monto Promedio (log)",
            title="Monto Promedio (log) por País Destino",
            color_discrete_sequence=[COLORS["primary"]],
            template="plotly_white", height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Correlaciones ────────────────────────────────────────────────
    with tab4:
        num_cols = ["Amount_log", "Risk_Score", "Shell Companies Involved",
                    "Reported_bin", "High_Risk", "Hour", "Month"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Matriz de Correlación", height=480,
                        template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# MÓDULO 2 — Clasificación

elif module == "Clasificación":
    st.markdown('<span class="section-badge">🎯 Clasificación</span>',
                unsafe_allow_html=True)
    st.markdown("**Variable objetivo:** `High_Risk` — 1 si Risk Score ≥ 7, 0 en caso contrario.")

    # ── Sidebar específico ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuración Clasificación")
        all_clfs = list(get_classifiers().keys())
        sel_models = st.multiselect("Modelos", all_clfs, default=all_clfs[:4])
        cutoff = st.slider("Probabilidad de Corte", 0.01, 0.99, 0.50, 0.01,
                           help="Umbral que convierte probabilidades en clases predichas.")
        use_smote = st.toggle("Aplicar SMOTE (balanceo)", value=True)

    st.markdown('<div class="info-box"> <b>Probabilidad de Corte:</b> En datasets desbalanceados, '
                'ajustar el umbral de decisión puede mejorar significativamente el Recall '
                'de la clase minoritaria sin sacrificar demasiado Precision.</div>',
                unsafe_allow_html=True)

    if not sel_models:
        st.warning("Selecciona al menos un modelo en el panel lateral.")
        st.stop()

    # ── Balance de clases ──────────────────────────────────────────────────
    with st.expander("Balance de clases antes del SMOTE"):
        c1, c2 = st.columns(2)
        vc = df["High_Risk"].value_counts()
        c1.metric("Clase 0 (Bajo Riesgo)", f"{vc.get(0,0):,}")
        c2.metric("Clase 1 (Alto Riesgo)", f"{vc.get(1,0):,}")
        fig = px.pie(values=vc.values, names=["Bajo Riesgo", "Alto Riesgo"],
                     color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]],
                     height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Entrenamiento ───────────────────────────────────────────────────────
    X, y, feat_names = get_Xy(df, task="classification")

    if st.button("Ejecutar Benchmarking de Clasificación", type="primary"):
        with st.spinner("Ejecutando K-Fold Cross Validation…"):
            metrics_df, fold_results = benchmark_classifiers(
                X, y, sel_models,
                n_splits=n_splits,
                cutoff=cutoff,
                use_smote=use_smote,
                random_state=int(random_seed),
            )
        st.session_state["clf_metrics"] = metrics_df
        st.session_state["clf_folds"]   = fold_results
        st.session_state["clf_feats"]   = feat_names

    if "clf_metrics" in st.session_state:
        metrics_df   = st.session_state["clf_metrics"]
        fold_results = st.session_state["clf_folds"]
        feat_names   = st.session_state["clf_feats"]

        st.markdown("---")
        st.markdown("#### 🏆 Tabla de Benchmarking")
        st.dataframe(
            styled_benchmark_table(metrics_df.set_index("Modelo")),
            use_container_width=True,
        )

        best_model = metrics_df.iloc[0]["Modelo"]
        st.success(f"Mejor modelo por AUC: **{best_model}** "
                   f"(AUC = {metrics_df.iloc[0]['AUC']:.4f})")

        tab_roc, tab_cm, tab_cut, tab_rep = st.tabs([
            "Curvas ROC", "Matriz de Confusión",
            "Análisis de Corte", "Reporte Detallado"
        ])

        with tab_roc:
            fig_roc = plot_roc_curves(fold_results)
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("""
            **Interpretación AUC:**
            | Rango | Calidad |
            |-------|---------|
            | 0.90 – 1.00 | Excelente |
            | 0.80 – 0.90 | Bueno |
            | 0.70 – 0.80 | Aceptable |
            | < 0.70 | Pobre |
            """)

        with tab_cm:
            best_data = fold_results[best_model]
            fig_cm = plot_confusion_matrix(
                best_data["y_true"], best_data["y_pred"],
                title=f"Matriz de Confusión — {best_model} (umbral={cutoff})"
            )
            st.pyplot(fig_cm)

        with tab_cut:
            best_data = fold_results[best_model]
            fig_cut, best_t = plot_cutoff_analysis(
                best_data["y_true"], best_data["y_prob"],
                title=f"Análisis de Corte — {best_model}"
            )
            st.plotly_chart(fig_cut, use_container_width=True)
            st.info(f"Umbral óptimo (mayor F1): **{best_t:.2f}** | "
                    f"Umbral actual seleccionado: **{cutoff:.2f}**")

        with tab_rep:
            best_data = fold_results[best_model]
            rep_df = classification_report_df(
                best_data["y_true"], best_data["y_pred"]
            )
            st.markdown(f"##### Reporte de Clasificación — {best_model}")
            st.dataframe(rep_df.style.format("{:.4f}", na_rep="—"),
                         use_container_width=True)

            # Feature importances si es tree-based
            if best_model in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.tree import DecisionTreeClassifier
                mdls = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
                }
                if best_model in mdls:
                    m = mdls[best_model]
                    m.fit(X, y)
                    fig_fi = plot_feature_importance(
                        m.feature_importances_, feat_names,
                        title=f"Importancia de Variables — {best_model}"
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

    else:
        st.info("Configura los parámetros y presiona **Ejecutar Benchmarking**.")


# MÓDULO 3 — Regresión
elif module == "Regresión":
    st.markdown('<span class="section-badge"> Regresión</span>',
                unsafe_allow_html=True)
    st.markdown("**Variable objetivo:** `Amount_log` — logaritmo del monto de la transacción.")

    with st.sidebar:
        st.markdown("### Configuración Regresión")
        all_regs = list(get_regressors().keys())
        sel_regs = st.multiselect("Modelos", all_regs, default=all_regs[:4])

    if not sel_regs:
        st.warning("Selecciona al menos un modelo.")
        st.stop()

    X_r, y_r, feat_names_r = get_Xy(df, task="regression")

    if st.button("Ejecutar Benchmarking de Regresión", type="primary"):
        with st.spinner("Ejecutando K-Fold para Regresión…"):
            metrics_df_r, oof_results = benchmark_regressors(
                X_r, y_r, sel_regs,
                n_splits=n_splits,
                random_state=int(random_seed),
            )
        st.session_state["reg_metrics"] = metrics_df_r
        st.session_state["reg_oof"]     = oof_results
        st.session_state["reg_feats"]   = feat_names_r

    if "reg_metrics" in st.session_state:
        metrics_df_r = st.session_state["reg_metrics"]
        oof_results  = st.session_state["reg_oof"]
        feat_names_r = st.session_state["reg_feats"]

        st.markdown("---")
        st.markdown("#### 🏆 Tabla de Benchmarking (Regresión)")
        st.dataframe(
            styled_benchmark_table(metrics_df_r.set_index("Modelo")),
            use_container_width=True,
        )

        best_reg = metrics_df_r.iloc[0]["Modelo"]
        st.success(f"Mejor modelo por R²: **{best_reg}** "
                   f"(R² = {metrics_df_r.iloc[0]['R²']:.4f})")

        tab_pv, tab_err, tab_fi = st.tabs([
            "Predicho vs Real", "Distribución de Errores", "📌 Importancia"
        ])

        with tab_pv:
            sel_show = st.selectbox("Modelo a visualizar", list(oof_results.keys()))
            data_r = oof_results[sel_show]
            # Mostrar solo 500 puntos para legibilidad
            n_show = min(500, len(data_r["y_true"]))
            fig_pv = plot_pred_vs_real(
                data_r["y_true"][:n_show],
                data_r["y_pred"][:n_show],
                title=f"Real vs Predicho — {sel_show} (primeros {n_show} pts)"
            )
            st.plotly_chart(fig_pv, use_container_width=True)

            # Scatter plot
            fig_sc = px.scatter(
                x=data_r["y_true"][:n_show], y=data_r["y_pred"][:n_show],
                labels={"x": "Real", "y": "Predicho"},
                title="Scatter: Real vs Predicho",
                opacity=0.5,
                color_discrete_sequence=[COLORS["secondary"]],
                template="plotly_white", height=400,
            )
            mn = min(data_r["y_true"].min(), data_r["y_pred"].min())
            mx = max(data_r["y_true"].max(), data_r["y_pred"].max())
            fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                        line=dict(dash="dash", color="grey"),
                                        name="Línea perfecta"))
            st.plotly_chart(fig_sc, use_container_width=True)

        with tab_err:
            data_r = oof_results[best_reg]
            residuals = data_r["y_true"] - data_r["y_pred"]
            fig_res = px.histogram(residuals, nbins=60,
                                   title=f"Distribución de Residuos — {best_reg}",
                                   color_discrete_sequence=[COLORS["primary"]],
                                   template="plotly_white", height=380)
            fig_res.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)

        with tab_fi:
            importances = get_feature_importances(best_reg, X_r, y_r)
            if importances is not None:
                fig_fi = plot_feature_importance(
                    importances, feat_names_r,
                    title=f"Importancia de Variables — {best_reg}"
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("El modelo seleccionado no expone importancia de variables.")

    else:
        st.info("Configura los parámetros y presiona **Ejecutar Benchmarking**.")


# MÓDULO 4 — Series de Tiempo
elif module == "Series de Tiempo":
    st.markdown('<span class="section-badge">Series de Tiempo</span>',
                unsafe_allow_html=True)
    st.markdown("Predicción del **monto promedio diario de transacciones** (log-escala).")

    with st.sidebar:
        st.markdown("### Configuración Series de Tiempo")
        freq_map = {"Diaria": "D", "Semanal": "W", "Mensual": "ME"}
        freq_label = st.selectbox("Frecuencia de agregación", list(freq_map.keys()))
        freq = freq_map[freq_label]

        sp = st.slider("Periodos estacionales (HW)", 2, 30, 7)
        nn_epochs = st.slider("Épocas LSTM", 10, 100, 30)

        all_ts_methods = [
            "Holt-Winters",
            "Holt-Winters Calibrado",
            "ARIMA",
            "ARIMA Calibrado",
            "Deep Learning (LSTM)",
        ]
        sel_ts = st.multiselect("Métodos", all_ts_methods, default=all_ts_methods[:4])

    ts = get_timeseries(df, freq=freq)

    # Visualizar serie completa
    st.markdown("#### Serie de Tiempo completa")
    fig_ts_full = go.Figure()
    fig_ts_full.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines", line=dict(color=COLORS["primary"], width=1.5),
        name="Monto log"
    ))
    fig_ts_full.update_layout(
        title=f"Monto Promedio de Transacciones ({freq_label})",
        xaxis_title="Fecha", yaxis_title="Amount (log)",
        height=380, template="plotly_white",
    )
    st.plotly_chart(fig_ts_full, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Observaciones", f"{len(ts):,}")
    col2.metric("Inicio", str(ts.index.min().date()))
    col3.metric("Fin",    str(ts.index.max().date()))

    st.markdown("---")

    if not sel_ts:
        st.warning("Selecciona al menos un método en el panel lateral.")
        st.stop()

    if st.button("Ejecutar Benchmarking de Series de Tiempo", type="primary"):
        with st.spinner("Entrenando modelos de series de tiempo… (puede tardar ~1-2 min)"):
            metrics_ts, forecasts_ts, extra_ts = benchmark_timeseries(
                ts,
                test_size=test_size,
                seasonal_periods=sp,
                selected_methods=sel_ts,
                nn_epochs=nn_epochs,
            )
        st.session_state["ts_metrics"]   = metrics_ts
        st.session_state["ts_forecasts"] = forecasts_ts
        st.session_state["ts_extra"]     = extra_ts

    if "ts_metrics" in st.session_state:
        metrics_ts   = st.session_state["ts_metrics"]
        forecasts_ts = st.session_state["ts_forecasts"]
        extra_ts     = st.session_state["ts_extra"]

        st.markdown("#### Tabla de Benchmarking (Series de Tiempo)")
        st.dataframe(
            styled_benchmark_table(metrics_ts.set_index("Modelo")),
            use_container_width=True,
        )

        best_ts = metrics_ts.dropna(subset=["RMSE"]).iloc[0]["Modelo"]
        st.success(f"Mejor método por RMSE: **{best_ts}**")

        if best_ts in extra_ts:
            st.info(f"{extra_ts[best_ts]}")

        # Gráfico comparativo de todas las predicciones
        st.markdown("#### Comparación Visual de Pronósticos")
        fig_comp = go.Figure()

        # Train
        tr_idx, tr_vals = forecasts_ts["train"]
        fig_comp.add_trace(go.Scatter(
            x=tr_idx, y=tr_vals,
            mode="lines", name="Entrenamiento",
            line=dict(color=COLORS["primary"], width=2),
        ))

        # Real test
        te_idx, te_vals = forecasts_ts["test"]
        fig_comp.add_trace(go.Scatter(
            x=te_idx, y=te_vals,
            mode="lines", name="Real (Test)",
            line=dict(color="#2c3e50", width=2, dash="dash"),
        ))

        # Predicciones
        palette = px.colors.qualitative.Set2
        for i, method in enumerate(sel_ts):
            if method in forecasts_ts:
                pred_idx, pred_vals = forecasts_ts[method]
                fig_comp.add_trace(go.Scatter(
                    x=pred_idx, y=pred_vals,
                    mode="lines", name=f"Pred: {method}",
                    line=dict(color=palette[i % len(palette)], width=2),
                ))

        # Línea divisoria train/test
        fig_comp.add_vline(x=te_idx[0].isoformat(), line_dash="dot", line_color="grey",
                           annotation_text="Train|Test")

        fig_comp.update_layout(
            title="Comparación de Métodos — Zona de Prueba",
            xaxis_title="Fecha", yaxis_title="Amount (log)",
            height=500, template="plotly_white",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Métricas individuales por método
        st.markdown("#### Métricas por Método")
        for _, row in metrics_ts.iterrows():
            with st.expander(f"{row['Modelo']}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE",      f"{row['MAE']:.4f}"     if pd.notna(row['MAE'])      else "N/A")
                c2.metric("RMSE",     f"{row['RMSE']:.4f}"    if pd.notna(row['RMSE'])     else "N/A")
                c3.metric("MAPE (%)", f"{row['MAPE (%)']:.2f}"if pd.notna(row['MAPE (%)']) else "N/A")

                if row["Modelo"] in forecasts_ts:
                    p_idx, p_vals = forecasts_ts[row["Modelo"]]
                    te_idx2, te_vals2 = forecasts_ts["test"]
                    fig_m = plot_pred_vs_real(
                        te_vals2[:len(p_vals)], p_vals,
                        index=p_idx,
                        title=f"Real vs Predicho — {row['Modelo']}",
                        is_ts=True,
                    )
                    st.plotly_chart(fig_m, use_container_width=True)

    else:
        st.info("Configura los parámetros y presiona **Ejecutar Benchmarking**.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#95a5a6; font-size:0.8rem;'>"
    "BCD-7213 · Minería de Datos Avanzada · Universidad LEAD · I Cuatrimestre 2026 · "
    "Dr. Juan Murillo-Morera"
    "</p>",
    unsafe_allow_html=True
)