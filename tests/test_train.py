import pandas as pd
import numpy as np
from pathlib import Path

from credit_scoring.train import TrainConfig, train_once

def _synth_df(n=600, seed=123):
    rng = np.random.default_rng(seed)
    income = rng.uniform(80_000, 200_000, size=n)
    credit = rng.uniform(50_000, 400_000, size=n)
    annuity = credit / rng.uniform(8, 20, size=n)
    ext1 = rng.beta(5, 2, size=n)
    ext2 = rng.beta(4, 3, size=n)
    ext3 = rng.beta(6, 2, size=n)
    days_birth = -rng.integers(25, 65, size=n) * 365
    days_employed = -rng.integers(0, 20, size=n) * 365
    # qq anomalies
    idx = rng.choice(n, size=max(1, n // 20), replace=False)
    days_employed[idx] = 365243

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

    # y dépend des ext + ratio
    ratio = credit / np.maximum(income, 1.0)
    risk = (1 - (ext1 + ext2 + ext3) / 3) * 0.7 + np.clip(ratio, 0, 5) * 0.3
    p = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)
    y = (p > np.quantile(p, 0.7)).astype(int)
    df["TARGET"] = y
    return df

def test_train_once_writes_artifacts(tmp_path: Path):
    df = _synth_df(n=500, seed=0)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    cfg = TrainConfig(
        data_path=csv_path,
        target_col="TARGET",
        algorithm="logistic",
        calibrate=True,
        test_size=0.25,
        random_state=0,
        fn_cost=5.0,
        fp_cost=1.0,
        sample_frac=0.5,  # accélère le test
        model_path=tmp_path / "models" / "model.joblib",
        metrics_path=tmp_path / "artifacts" / "metrics.json",
    )
    metrics = train_once(cfg)

    # artefacts présents
    assert cfg.model_path.exists()
    assert cfg.metrics_path.exists()

    # métriques cohérentes
    assert 0.0 <= metrics["threshold"] <= 1.0
    assert metrics["valid"]["auc"] >= 0.65
    assert 0.0 <= metrics["valid"]["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["valid"]["business_score"] <= 1.0

    # feature list enregistrée
    assert metrics["feature_count"] > 0
    assert isinstance(metrics["feature_columns"], list) and len(metrics["feature_columns"]) == metrics["feature_count"]
