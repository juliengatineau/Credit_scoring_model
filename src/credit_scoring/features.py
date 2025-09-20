"""
Feature engineering minimal pour le scoring.

- `safe_div` : division élémentaire robuste (évite les divisions par zéro).
- `build_features` : ajoute des ratios et flags utiles au modèle.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Divise `a` par `b` élément par élément.

    - Si `b == 0`, renvoie NaN (au lieu d'inf).
    - Conserve le dtype float pour permettre les NaN.
    - Hypothèse : `a` et `b` sont broadcastables.

    Parameters
    ----------
    a : np.ndarray
        Numérateur.
    b : np.ndarray
        Dénominateur.

    Returns
    -------
    np.ndarray
        Tableau de même forme que le broadcast de `a` et `b`, avec NaN là où b==0.
    """
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    out = np.full(np.broadcast(a, b).shape, np.nan, dtype=float)  # résultat float rempli de NaN
    mask = b != 0
    # np.divide gère le broadcast proprement
    out[mask] = np.divide(a[mask], b[mask])
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un ensemble minimal de features dérivées à partir d'un DataFrame brut.

    Règles:
    - Remplit les colonnes clés manquantes avec NaN pour rester tolérant aux schémas incomplets.
    - Crée des ratios robustes :
        * credit_income_pct  = AMT_CREDIT / AMT_INCOME_TOTAL
        * annuity_income_pct = AMT_ANNUITY / AMT_INCOME_TOTAL
        * ext_source_mean    = moyenne de EXT_SOURCE_1..3 (ignorant les NaN)
    - Gère l'anomalie Home Credit :
        * DAYS_EMPLOYED > 0 -> flag DAYS_EMPLOYED_ANOM = 1 et remplace la valeur par NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Données brutes d'une ou plusieurs demandes de prêt.

    Returns
    -------
    pd.DataFrame
        Copie de `df` enrichie des colonnes d'ingénierie de features.
    """
    X = df.copy()

    # Colonnes minimales attendues
    needed = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    for col in needed:
        if col not in X.columns:
            X[col] = np.nan  # tolérance au schéma

    # Ratios robustes (évite divisions par 0)
    X["credit_income_pct"] = safe_div(
        X["AMT_CREDIT"].to_numpy(float),
        X["AMT_INCOME_TOTAL"].to_numpy(float),
    )
    X["annuity_income_pct"] = safe_div(
        X["AMT_ANNUITY"].to_numpy(float),
        X["AMT_INCOME_TOTAL"].to_numpy(float),
    )

    # Moyenne des ext sources (NaN-safe)
    X["ext_source_mean"] = np.nanmean(
        np.vstack(
            [
                X["EXT_SOURCE_1"].to_numpy(float),
                X["EXT_SOURCE_2"].to_numpy(float),
                X["EXT_SOURCE_3"].to_numpy(float),
            ]
        ),
        axis=0,
    )

    # Flag anomalie 1000 ans (DAYS_EMPLOYED positif) + nettoyage
    X["DAYS_EMPLOYED_ANOM"] = (X["DAYS_EMPLOYED"] > 0).astype(float)
    X.loc[X["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = np.nan

    return X
