# src/credit_scoring/train.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from credit_scoring.preprocess import make_preprocessor, transform_to_frame
from credit_scoring.metrics import find_optimal_threshold, compute_metrics  # tu as déjà ces fonctions


@dataclass
class TrainConfig:
    data_path: Path
    target_col: str
    algorithm: str
    sample_frac: float
    random_state: int
    name: str
    models_dir: Path
    artifacts_dir: Path


# ----------------- utils modèles -----------------
def _make_estimator(algo: str, *, random_state: int, scale_pos_weight: float):
    a = algo.lower()
    if a == "logistic":
        return LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )
    if a == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            min_samples_split=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
    if a == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception as e:
            raise ImportError("Installe xgboost: `poetry add xgboost`") from e
        return XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
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
    if a == "lgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception as e:
            raise ImportError("Installe lightgbm: `poetry add lightgbm`") from e
        return LGBMClassifier(
            n_estimators=700,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary",
            n_jobs=-1,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
    raise ValueError(f"Algo inconnu: {algo}")


def _compute_global_shap_fast_on_pipeline(
    pipe: Pipeline, df_train: pd.DataFrame, feature_names: list[str], max_samples: int = 1500
) -> tuple[dict[str, float], int]:
    """
    Retourne (importances_moy_abs_par_feature, n_samples_utilisés)
    importances_moy_abs_par_feature : {feature -> mean(|SHAP|)}
    """
    try:
        import shap  # type: ignore
    except Exception:
        return {}, 0

    # sous-échantillon
    if len(df_train) > max_samples:
        df_sample = df_train.sample(n=max_samples, random_state=0)
    else:
        df_sample = df_train

    # transforme vers X numérique aligné
    X = transform_to_frame(pipe.named_steps["prep"], feature_names, df_sample)
    clf = pipe.named_steps["clf"]
    name = clf.__class__.__name__.lower()

    try:
        # Essaye d’utiliser l’explainer le plus adapté
        if any(k in name for k in ["xgb", "lgbm", "randomforest", "forest", "tree", "boost"]):
            explainer = shap.TreeExplainer(clf)  # rapide pour arbres/boosting
            shap_vals = explainer.shap_values(X.values)
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        elif "logisticregression" in name:
            explainer = shap.LinearExplainer(clf, X.values)
            sv = explainer.shap_values(X.values)
        else:
            # fallback (permutation) robuste
            f = lambda Z: clf.predict_proba(Z)[:, 1]
            explainer = shap.Explainer(f, X.values, algorithm="permutation")
            sv = explainer(X.values).values
    except Exception:
        return {}, len(df_sample)

    mean_abs = np.abs(sv).mean(axis=0)
    out = {feat: float(val) for feat, val in zip(feature_names, mean_abs)}
    return out, len(df_sample)


# ----------------- chargement + train -----------------
def _load_df(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    if cfg.sample_frac < 1.0:
        df = df.sample(frac=cfg.sample_frac, random_state=cfg.random_state)
    return df


def train_once(cfg: TrainConfig) -> Dict[str, Any]:
    df = _load_df(cfg)
    y = df[cfg.target_col].astype(int)

    # Préprocesseur (fit sur le train full) + noms de features
    prep_bundle = make_preprocessor(df.drop(columns=[cfg.target_col], errors="ignore"))
    prep = prep_bundle.pipeline
    feature_names = prep_bundle.feature_names

    # Déséquilibre -> scale_pos_weight pour boosting
    pos = float(y.sum())
    neg = float(len(y) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # Modèle
    clf = _make_estimator(cfg.algorithm, random_state=cfg.random_state, scale_pos_weight=scale_pos_weight)

    # Pipeline complet: preprocess + clf
    pipe = Pipeline([
        ("prep", prep),
        ("clf", clf),
    ])

    # Fit
    X_raw = df.drop(columns=[cfg.target_col], errors="ignore")
    pipe.fit(X_raw, y)
    print("--- modèle entraîné ---")

    # Probabilités + seuil optimal métier
    proba = pipe.predict_proba(X_raw)[:, 1]
    thr = float(find_optimal_threshold(y.values, proba, fn_cost=10.0, fp_cost=1.0))
    y_pred = (proba >= thr).astype(int)

    # Métriques
    metrics = compute_metrics(y.values, y_pred, proba, threshold=thr)
    print("--- metrics calculées ---")
    
    # SHAP global (pré-calculé une fois, format standard pour l'app)
    imp_dict, n_used = _compute_global_shap_fast_on_pipeline(pipe, X_raw, feature_names)
    if imp_dict:
        items = sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_n = min(30, len(items))
        payload = {
            "features": [k for k, _ in items[:top_n]],
            "mean_abs_shap": [float(v) for _, v in items[:top_n]],
            "n_sample": int(n_used),
            "top_n": int(top_n),
            "model_class": pipe.named_steps["clf"].__class__.__name__,
            "explainer": "auto(tree/linear/permutation)",
            "generated_by": "train-scoring",
        }
        with open(cfg.artifacts_dir / f"{cfg.name}_global_shap.json", "w") as f:
            json.dump(payload, f, indent=2)


# ----------------- CLI -----------------
def main() -> None:
    import argparse

    p = argparse.ArgumentParser("Train credit scoring model (industrial preprocessor)")
    p.add_argument("--data", dest="data_path", type=Path, required=True)
    p.add_argument("--target", dest="target_col", type=str, default="TARGET")
    p.add_argument("--algorithm", type=str, choices=["logistic", "rf", "xgb", "lgbm"], default="logistic")
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--name", type=str, default="model_nb")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        target_col=args.target_col,
        algorithm=args.algorithm,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
        name=args.name,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
    )
    out = train_once(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
