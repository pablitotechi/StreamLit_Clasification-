# BCD-7213 · Minería de Datos Avanzada — Caso de Estudio #1
**Universidad LEAD · I Cuatrimestre 2026 · Dr. Juan Murillo-Morera**

## Descripción
Aplicación Streamlit para benchmarking de modelos de **Clasificación**, **Regresión** y **Series de Tiempo** aplicados al dataset *Big Black Money Dataset* (transacciones financieras de lavado de dinero).

---

## Estructura del proyecto

```
project/
├── app.py                  ← Aplicación Streamlit principal
├── requirements.txt
├── README.md
├── data/
│   └── Big_Black_Money_Dataset.csv
└── src/
    ├── io.py               ← Carga del dataset
    ├── prep.py             ← Limpieza y preprocesamiento
    ├── classification.py   ← Modelos de clasificación + SMOTE + K-Fold + AUC
    ├── regression.py       ← Modelos de regresión + K-Fold
    ├── timeseries.py       ← HW, HW-Cal, ARIMA, ARIMA-Cal, LSTM
    └── utils.py            ← Visualizaciones compartidas (ROC, confusión, etc.)
```

---

## Instalación y ejecución

### 1. Clonar / descomprimir el proyecto
```bash
cd project/
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
streamlit run app.py
```
Abre tu navegador en `http://localhost:8501`

---

## Módulos de la aplicación

| Módulo | Descripción |
|--------|-------------|
| 📊 Exploración | EDA interactivo: distribuciones, geografía, correlaciones |
| 🎯 Clasificación | 7 modelos · K-Fold · SMOTE · Umbral de corte · ROC/AUC · Visualización de balance de clases |
| 📈 Regresión | 9 modelos · K-Fold · R², MAE, RMSE, MAPE |
| 📉 Series de Tiempo | HW · HW-Cal · ARIMA · ARIMA-Cal · LSTM |

---

## Conceptos implementados

- **Probabilidad de Corte (cutoff):** slider interactivo que ajusta el umbral de decisión para clases desbalanceadas
- **K-Fold Cross Validation:** `StratifiedKFold` (clasificación) y `KFold` (regresión) configurable desde la UI
- **AUC (Area Under the Curve):** curvas ROC superpuestas con Plotly para comparar modelos visualmente
- **SMOTE:** balanceo de clases minoritarias dentro de cada fold para evitar data leakage, con visualización del balance antes y después
- **ARIMA Calibrado:** grid search sobre (p,d,q) minimizando AIC con test ADF para determinar d
- **LSTM:** red neuronal recurrente con predicción iterativa multi-paso

---

## Dependencias principales
- `streamlit` — interfaz gráfica
- `scikit-learn` — modelos ML clásicos
- `imbalanced-learn` — SMOTE
- `statsmodels` — Holt-Winters, ARIMA
- `tensorflow/keras` — Red neuronal LSTM
- `plotly` — visualizaciones interactivas
