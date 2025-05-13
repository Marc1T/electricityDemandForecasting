# src/models/modeling.py

import joblib
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import pmdarima as pm

from sklearn.linear_model    import Ridge, ElasticNet
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.models.evaluation import evaluate_regression
from src.models.tuning     import tune_lightgbm


def load_processed(zone: str, horizon: str, processed_dir: Path) -> pd.DataFrame:
    """
    Charge le DataFrame 'processed' pour la zone et l’horizon ('hourly' ou 'daily').
    """
    if horizon == "daily":
        suffix = "_processed_daily.parquet"
    else:  # hourly
        suffix = "_processed_hourly.parquet"
    file = processed_dir / f"{zone}{suffix}"
    return pd.read_parquet(file)


def train_models_for_zone(zone: str,
                          processed_dir: Path,
                          models_dir: Path) -> dict:
    """
    Pour la zone donnée, entraîne et sauvegarde 4 modèles sur deux horizons:
      - ElasticNet, Ridge, RandomForest, auto_arima
    Renvoie un dict {horizon: {model_name: metrics_dict}}.
    """
    results = {}
    for horizon in ["daily", "hourly"]:
        # 1. Charger les données
        df = load_processed(zone, horizon, processed_dir)
        target   = "demand"
        features = [c for c in df.columns if c != target]

        # 2. Split chronologique 80/20
        split = int(len(df) * 0.8)
        X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
        y_train, y_test = df[target].iloc[:split], df[target].iloc[split:]

        metrics = {}

        # 3. ElasticNet avec GridSearchCV temporel
        enet = ElasticNet(max_iter=10000)
        param_grid = {
            "alpha":    [0.1, 1.0, 10.0],
            "l1_ratio": [0.2, 0.5, 0.8]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        grid_en = GridSearchCV(
            enet, param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        grid_en.fit(X_train, y_train)
        best_enet = grid_en.best_estimator_
        metrics["ElasticNet"] = evaluate_regression(best_enet, X_test, y_test)
        joblib.dump(best_enet, models_dir/f"{zone}_{horizon}_elasticnet.pkl")

        # 4. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        metrics["Ridge"] = evaluate_regression(ridge, X_test, y_test)
        joblib.dump(ridge, models_dir/f"{zone}_{horizon}_ridge.pkl")

        # 5. RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        metrics["RandomForest"] = evaluate_regression(rf, X_test, y_test)
        joblib.dump(rf, models_dir/f"{zone}_{horizon}_rf.pkl")

        # # 6. LGBM optimisé (facultatif)
        # lgbm = tune_lightgbm(X_train, y_train)
        # metrics["LightGBM"] = evaluate_regression(lgbm, X_test, y_test)
        # joblib.dump(lgbm, models_dir/f"{zone}_{horizon}_lgbm.pkl")

        # 7. auto_arima (pmdarima) avec saisonnalité
        #    s=7 pour daily, s=24 pour hourly
        m = 7 if horizon == "daily" else 24
        if horizon == "hourly":
            metrics["auto_arima"] = {"MAE": float('inf'), "RMSE": float('inf'), "MAPE": float('inf')}
        else:
            y_train_arima = y_train
            y_test_arima = y_test

            arima_model = pm.auto_arima(
                y_train_arima,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None, D=None,
                seasonal=True, m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            preds = arima_model.predict(n_periods=len(y_test_arima))
            pred_series = pd.Series(preds, index=y_test_arima.index)
            metrics["auto_arima"] = evaluate_regression(
                arima_model, None, y_test_arima, pred_series=pred_series
            )
            joblib.dump(arima_model, models_dir/f"{zone}_{horizon}_auto_arima.pkl")

        results[horizon] = metrics

    return results