# src/models/tuning.py

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def tune_lightgbm(X, y):
    """
    Recherche d’hyperparamètres pour un LGBMRegressor via GridSearchCV
    sur un split temporel.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [5, 10, None],
        "learning_rate":[0.01, 0.1]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        LGBMRegressor(random_state=42),
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid.fit(X, y)
    return grid.best_estimator_
