from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_logistic_pipeline(random_state: int = 42) -> Pipeline:
    """
    Pipeline régression logistique.
    - Standardise les features.
    - class_weight='balanced' pour gérer le déséquilibre.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=200,
                n_jobs=-1,
                class_weight="balanced",
                random_state=random_state
            )),
        ]
    )


def build_rf_pipeline(random_state: int = 42) -> Pipeline:
    """
    Pipeline RandomForest.
    - Pas de scaling nécessaire.
    - class_weight='balanced_subsample' pour limiter le biais de classe.
    """
    return Pipeline(
        steps=[
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state,
            )),
        ]
    )


def build_xgb_pipeline(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Pipeline XGBoost (binary:logistic).
    - scale_pos_weight = (#neg / #pos) pour déséquilibre.
    - tree_method='hist' pour vitesse CPU.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as e:
        raise ImportError(
            "xgboost est requis. Installe: `poetry add xgboost`"
        ) from e

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        missing=np.nan,
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline([("clf", clf)])


def build_lgbm_pipeline(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Pipeline LightGBM (objective=binary).
    - scale_pos_weight = (#neg / #pos) pour déséquilibre.
    """
    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except Exception as e:
        raise ImportError(
            "lightgbm est requis. Installe: `poetry add lightgbm`"
        ) from e

    clf = LGBMClassifier(
        n_estimators=600,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=-1,
        random_state=random_state,
        # Tu peux aussi utiliser class_weight, mais on reste cohérents avec scale_pos_weight.
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline([("clf", clf)])
