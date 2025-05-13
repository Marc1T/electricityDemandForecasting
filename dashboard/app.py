# dashboard/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model as load_keras
from src.utils.best_model_utils import load_best_table, get_model_path, load_best_model
from src.models.dl_utils import create_sequences, scale_data

st.set_page_config(layout="wide", page_title="Energy Demand Dashboard")

# 1. Chargement des métadonnées
best_df = load_best_table()
zones   = best_df['zone'].tolist()
horizons = ["daily", "hourly"]

# 2. Sidebar de sélection
st.sidebar.title("Configuration")
zone    = st.sidebar.selectbox("Zone", zones)
horizon = st.sidebar.selectbox("Horizon", horizons)

# 3. Chargement des données processed
cfg = pd.read_yaml("config.yaml")
proc_dir = Path(cfg["paths"]["data"]["processed"])
file = proc_dir / f"{zone}_processed_{horizon}.parquet"
df = pd.read_parquet(file)

st.title(f"Prévisions de la demande — {zone} ({horizon})")

# 4. Afficher quelques stats
st.subheader("Statistiques de base")
st.write(df["demand"].describe())

# 5. Charger le meilleur modèle
model = load_best_model(zone, horizon)
st.subheader("Modèle sélectionné")
st.write(model.__class__.__name__)

# 6. Préparer X_test et y_test (dernier 20%)
split = int(len(df) * 0.8)
test_df = df.iloc[split:].copy()
X_test, y_test = test_df.drop(columns="demand"), test_df["demand"]

# Si c'est un modèle DL, on doit créer des séquences
if hasattr(model, "predict") and isinstance(model, (type(load_keras()))):
    LOOKBACK = 24 if horizon=="hourly" else 7  # ajustable
    X_seq, y_seq = create_sequences(df, "demand", list(X_test.columns), LOOKBACK)
    # On prend la partie de test
    n_test = len(y_seq) - len(test_df)  # aligner
    X_test_seq = X_seq[-len(test_df):]
    # scale features
    X_test_seq, _, _ = scale_data(X_seq[: -len(test_df)], X_test_seq)
    y_pred = model.predict(X_test_seq).ravel()
    y_true = y_seq[-len(test_df):].ravel()
else:
    # modèle sklearn ou ARIMA
    # ARIMA peut renvoyer un Pandas Series
    try:
        y_pred = model.predict(X_test)
    except:
        y_pred = model.predict(n_periods=len(test_df))
    y_true = y_test.values

# 7. Affichage des courbes
st.subheader("Réel vs Prédit")
chart_df = pd.DataFrame({
    "réel": y_true,
    "prévision": y_pred
}, index=test_df.index)
st.line_chart(chart_df)

# 8. KPI de performance
from src.models.evaluation import evaluate_regression
kpi = evaluate_regression(None, None, pd.Series(y_true), pd.Series(y_pred))
st.subheader("Performance")
st.metric("MAE", f"{kpi['MAE']:.1f}")
st.metric("RMSE", f"{kpi['RMSE']:.1f}")
st.metric("MAPE", f"{kpi['MAPE']:.1f}%")
