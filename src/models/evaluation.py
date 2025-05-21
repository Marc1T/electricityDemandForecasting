# src/models/evaluation.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_regression(model, X_test, y_test, pred_series=None):
    """
    Calcule MAE, RMSE, MAPE pour un modèle sklearn ou SARIMAX.
    - model       : un objet sklearn ou statsmodels (SARIMAXResults)
    - X_test      : DataFrame X_test (None si pred_series fourni)
    - y_test      : Series y_test
    - pred_series : Series de prédictions (optionnel)
    """
    if pred_series is None:
        y_pred = model.predict(X_test)
    else:
        y_pred = pred_series

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
