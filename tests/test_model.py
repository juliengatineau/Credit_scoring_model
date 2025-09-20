import numpy as np
import pandas as pd
from pathlib import Path

from credit_scoring.features import build_features
from credit_scoring.model import (
    fit_pipeline,
    predict_proba,
    predict_with_threshold,
    evaluate_pipeline,
    choose_threshold,
    save_model,
    load_model,
)


def _toy_raw_df(n=400, seed=123):
    """
    Génère un petit dataset synthétique *compatible* avec build_features :
    - Colonnes AMT_* et EXT_SOURCE_*.
    - y dépend de ext_source_mean et de credit_income_pct -> séparabilité raisonnable.
    """
    rng = np.random.default_rng(seed)
    income = rng.uniform(80_000, 200_000, size=n)
    credit = rng.uniform(50_000, 400_000, size=n)
    annuity = credit / rng.uniform(8, 20, size=n)

    # EXT sources: mieux pour "bons" payeurs
    ext1 = rng.beta(5, 2, size=n)
    ext2 = rng.beta(4, 3, size=n)
    ext3 = rng.beta(6, 2, size=n)

    days_birth = -rng.integers(25, 65, size=n) * 365
    # 5% anomalies DAYS_EMPLOYED positif
    days_employed = -rng.integers(0, 20, size=n) * 365
    anom_idx = rng.choice(n, size=max(1, n // 20), replace=False)
    days_employed[anom_idx] = 365243

    df = pd.DataFrame({
        "AMT_CREDIT": credit,
        "AMT_INCOME_TOTAL": income,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    })

    X = build_features(df)
    # Génère y: plus de défaut si ext_mean bas et ratio crédit/revenu haut
    risk = (1 - X["ext_source_mean"].to_numpy(float)) * 0.7 + np.clip(X["credit_income_pct"].to_numpy(float), 0, 5) * 0.3
    p = np.clip(risk / (risk.max() + 1e-9), 0, 1)
    y = (p > np.quantile(p, 0.7)).astype(int)  # ~30% positifs
    return X.to_numpy(float), y


def test_fit_evaluate_logistic(tmp_path: Path):
    X, y = _toy_raw_df(n=500, seed=0)
    model = fit_pipeline(X, y, algorithm="logistic", calibrate=True, random_state=0)
    res = evaluate_pipeline(model, X, y, fn_cost=5.0, fp_cost=1.0)
    assert 0.0 <= res.threshold <= 1.0
    assert 0.0 <= res.balanced_accuracy <= 1.0
    assert 0.0 <= res.business_score <= 1.0
    # un minimum de performance attendu
    assert res.auc >= 0.75


def test_predict_shapes_and_threshold():
    X, y = _toy_raw_df(n=300, seed=1)
    model = fit_pipeline(X, y, algorithm="logistic", calibrate=True, random_state=1)
    proba = predict_proba(model, X)
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0) & (proba <= 1))

    thr = choose_threshold(y, proba, fn_cost=10.0, fp_cost=1.0, grid_size=501)
    yhat = predict_with_threshold(model, X, thr)
    assert set(np.unique(yhat)).issubset({0, 1})


def test_save_and_load_roundtrip(tmp_path: Path):
    X, y = _toy_raw_df(n=200, seed=2)
    model = fit_pipeline(X, y, algorithm="logistic", calibrate=False, random_state=2)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    assert path.exists()

    loaded = load_model(path)
    p1 = predict_proba(model, X)
    p2 = predict_proba(loaded, X)
    # pas nécessairement identiques bit-à-bit, mais proches
    assert np.allclose(p1, p2, atol=1e-8)
