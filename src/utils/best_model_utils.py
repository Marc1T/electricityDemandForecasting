# src/utils/best_model_utils.py

import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model as load_keras

BEST_CSV      = Path("outputs/reports/best_models.csv")
MODELS_DIR    = Path("models")
DL_MODELS_DIR = Path("models/dl")

def load_best_table() -> pd.DataFrame:
    """
    Charge le tableau best_models.csv indiquant
    pour chaque zone le meilleur modèle daily & hourly.
    """
    return pd.read_csv(BEST_CSV)

def get_model_path(zone: str, horizon: str, model_name: str) -> Path:
    """
    Construit le chemin vers le fichier du modèle sauvegardé.
    """
    model_name_clean = model_name.lower().replace("-", "")
    if model_name in ("ElasticNet","RandomForest","LightGBM","Ridge","ARIMA"):
        if model_name == "RandomForest":
            model_name_clean = "rf"
        filename = f"{zone}_{horizon}_{model_name_clean}.pkl"
        return MODELS_DIR/filename
    else:
        # LSTM, GRU, CNN, CNN-LSTM
        filename = f"{zone}_{horizon}_{model_name}.h5"
        return DL_MODELS_DIR/filename

def load_best_model(zone: str, horizon: str):
    """
    Lit best_models.csv, identifie et charge le modèle gagnant.
    """
    df = load_best_table()
    row = df[df.zone == zone].iloc[0]
    model_name = row[f"best_model_{horizon}"]
    path = get_model_path(zone, horizon, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {path}")
    if path.suffix == ".pkl":
        return joblib.load(path)
    elif path.suffix == ".h5":
        return load_keras(path)
    else:
        raise ValueError(f"Extension inconnue pour le modèle : {path.suffix}")
