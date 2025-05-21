from dotenv import load_dotenv
import os

load_dotenv()
os.chdir("..")

# dashboard/app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model as load_keras
from src.utils.best_model_utils import load_best_table, load_best_model, get_model_path
from src.models.dl_utils import create_sequences, scale_data
from src.models.evaluation import evaluate_regression

# Configuration de la page
st.set_page_config(layout="wide", page_title="Energy Demand Dashboard")

# 1. Chargement des métadonnées
best_df  = load_best_table()
zones    = best_df['zone'].tolist()
horizons = ["daily", "hourly"]

# 2. Sidebar de sélection
st.sidebar.title("Configuration")
zone    = st.sidebar.selectbox("Zone", zones)
horizon = st.sidebar.selectbox("Horizon", horizons)

# 3. Chargement des données processed
cfg        = pd.read_yaml("config.yaml")
proc_dir   = Path(cfg["paths"]["data"]["processed"])
file_path  = proc_dir / f"{zone}_processed_{horizon}.parquet"
df         = pd.read_parquet(file_path)

st.title(f"Prévisions de la demande — {zone.capitalize()} ({horizon})")

# 4. Statistiques descriptives
st.subheader("Statistiques de la série réelle")
st.write(df["demand"].describe())

# 5. Chargement du modèle gagnant
model = load_best_model(zone, horizon)
st.subheader("Modèle sélectionné")
st.write(type(model).__name__)

# 6. Préparation du jeu de test (dernier 20%)
split_idx = int(len(df) * 0.8)
test_df   = df.iloc[split_idx:].copy()
X_test    = test_df.drop(columns="demand")
y_true    = test_df["demand"].values

# 7. Prédiction selon le type de modèle
if isinstance(model, (type(load_keras("")),)):  # modèle Keras
    LOOKBACK = 24 if horizon=="hourly" else 7
    feature_cols = X_test.columns.tolist()
    X_full, y_full = create_sequences(df, "demand", feature_cols, LOOKBACK)
    # On aligne la fenêtre de test
    n_test = len(test_df)
    X_seq = X_full[-n_test:]
    _, X_seq_scaled, _ = scale_data(X_full[:-n_test], X_seq)
    y_pred = model.predict(X_seq_scaled).ravel()
    y_true = y_full[-n_test:].ravel()
else:
    # sklearn ou ARIMA
    try:
        y_pred = model.predict(X_test)
    except TypeError:
        # ARIMA: predict(n_periods)
        y_pred = model.predict(n_periods=len(test_df))
    y_pred = pd.Series(y_pred, index=test_df.index).values

# 8. Affichage des courbes
st.subheader("Comparaison Réel vs Prédiction")
chart_df = pd.DataFrame({
    "Réel":      y_true,
    "Prédiction": y_pred
}, index=test_df.index)
st.line_chart(chart_df)

# 9. KPI de performance (RMSE, MAE, MAPE)
kpi = evaluate_regression(None, None, pd.Series(y_true), pd.Series(y_pred))
st.subheader("Indicateurs de performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE",  f"{kpi['MAE']:.2f}")
col2.metric("RMSE", f"{kpi['RMSE']:.2f}")
col3.metric("MAPE", f"{kpi['MAPE']:.2f}%")
