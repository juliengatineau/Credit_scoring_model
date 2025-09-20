import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import safe_div, build_features


def test_safe_div_basic_and_zero_handling():
    a = np.array([10, 10, 10, 0], dtype=float)
    b = np.array([2, 0, 5, 0], dtype=float)
    out = safe_div(a, b)
    # 10/2=5, 10/0=NaN, 10/5=2, 0/0=NaN
    assert np.allclose(out[[0, 2]], [5.0, 2.0], equal_nan=True)
    assert np.isnan(out[1])
    assert np.isnan(out[3])


def test_build_features_adds_expected_columns_and_flags_anomaly():
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [200000, 500000],
            "AMT_INCOME_TOTAL": [100000, 250000],
            "AMT_ANNUITY": [20000, 40000],
            "DAYS_BIRTH": [-44 * 365, -31 * 365],
            "DAYS_EMPLOYED": [-3650, 365243],  # seconde ligne = anomalie
            "EXT_SOURCE_1": [0.6, 0.1],
            "EXT_SOURCE_2": [0.5, 0.2],
            "EXT_SOURCE_3": [0.7, 0.3],
        }
    )

    X = build_features(df)

    # Colonnes dérivées présentes
    for col in ["credit_income_pct", "annuity_income_pct", "ext_source_mean", "DAYS_EMPLOYED_ANOM"]:
        assert col in X.columns

    # Vérifie calculs simples
    assert np.isclose(X.loc[0, "credit_income_pct"], 200000 / 100000)
    assert np.isclose(X.loc[1, "credit_income_pct"], 500000 / 250000)

    # Moyenne ext sources
    assert np.isclose(X.loc[0, "ext_source_mean"], (0.6 + 0.5 + 0.7) / 3)
    assert np.isclose(X.loc[1, "ext_source_mean"], (0.1 + 0.2 + 0.3) / 3)

    # Anomalie : flag = 1 et remplacement par NaN
    assert X.loc[1, "DAYS_EMPLOYED_ANOM"] == 1.0
    assert pd.isna(X.loc[1, "DAYS_EMPLOYED"])

    # Cas normal : flag = 0
    assert X.loc[0, "DAYS_EMPLOYED_ANOM"] == 0.0


def test_build_features_tolerates_missing_columns():
    # on omet volontairement plusieurs colonnes
    df = pd.DataFrame({"AMT_CREDIT": [100000], "AMT_INCOME_TOTAL": [0]})  # division par 0 → NaN
    X = build_features(df)
    assert "credit_income_pct" in X.columns
    assert np.isnan(X.loc[0, "credit_income_pct"])  # 100000 / 0 → NaN
    # colonnes manquantes créées
    for col in ["AMT_ANNUITY", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_EMPLOYED"]:
        assert col in X.columns
