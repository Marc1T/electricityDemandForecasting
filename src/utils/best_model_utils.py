# src/utils/best_model_utils.py

import json
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model as load_keras

BEST_CSV = Path("outputs/reports/best_models.csv")
MODELS_DIR = Path("models")
DL_MODELS_DIR = Path("models/dl")

def load_best_table() -> pd.DataFrame:
    return pd.read_csv(BEST_CSV)

def get_model_path(zone: str, horizon: str, model_name: str) -> Path:
    """
    Retourne le chemin vers le fichier du modèle sauvegardé.
    horizon : 'daily' ou 'hourly'
    model_name : nom tel que dans best_models.csv
    """
    # extension joblib pour les classiques, .h5 pour DL
    if model_name in ("ElasticNet","RandomForest","LightGBM","Ridge","ARIMA"):
        filename = f"{zone}_{horizon}_{model_name.lower()}.pkl"
        return MODELS_DIR/filename
    else:
        # supposons les noms LSTM, GRU, CNN, CNN-LSTM
        filename = f"{zone}_{horizon}_{model_name}.h5"
        return DL_MODELS_DIR/filename

def load_best_model(zone: str, horizon: str):
    """
    Lit best_models.csv et charge l'objet modèle correspondant.
    """
    df = load_best_table()
    row = df[df.zone == zone].iloc[0]
    model_name = row[f"best_model_{horizon}"]
    path = get_model_path(zone, horizon, model_name)
    if path.suffix == ".pkl":
        return joblib.load(path)
    elif path.suffix == ".h5":
        return load_keras(path)
    else:
        raise ValueError(f"Extension inconnue pour {path}")
