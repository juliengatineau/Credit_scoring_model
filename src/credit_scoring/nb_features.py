# src/credit_scoring/nb_features.py
from __future__ import annotations

"""
Builder de features factorisé depuis les notebooks d'exploration/modélisation.

Objectif
--------
Transformer un DataFrame brut Home Credit en un DataFrame **numérique** prêt pour le modèle :
- Nettoyage des anomalies (DAYS_EMPLOYED == 365243)
- Ratios financiers (credit/income, annuity/income, term, goods/*)
- Agrégats des EXT_SOURCE_* (mean, sum/3)
- Variables de stabilité (registration/id_publish/phone_change/employed)
- Flags / propriétaires / documents (s’ils existent)
- Pas d'encodage one-hot ici (on reste simple et robuste)

Notes
-----
- On laisse des NaN lorsque nécessaire : l'imputation est faite plus tard dans le pipeline d'entraînement.
- Toutes les colonnes retournées sont numériques.

Exemple
-------
>>> from credit_scoring.nb_features import build_features_nb
>>> X = build_features_nb(df_raw)
"""

from typing import Iterable
import numpy as np
import pandas as pd

__all__ = ["build_features_nb"]


# ------------------ Helpers ------------------ #
def _safe_ratio(num: pd.Series | float, den: pd.Series | float) -> pd.Series:
    """
    Calcule num/den en évitant les divisions par zéro et les infinis.

    Retour
    ------
    pd.Series : série de float64 avec NaN si dénominateur nul/absent.
    """
    num_s = pd.to_numeric(num, errors="coerce") if isinstance(num, pd.Series) else pd.Series(num)
    den_s = pd.to_numeric(den, errors="coerce") if isinstance(den, pd.Series) else pd.Series(den)
    den_s = den_s.replace(0, np.nan)
    out = num_s.astype("float64") / den_s.astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _sum_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def _mean_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in df.columns if c in cols]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _abs_series(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce").abs() if name in df.columns else pd.Series(np.nan, index=df.index)


def _clip_percentile(s: pd.Series, hi: float = 99.0) -> pd.Series:
    if s.isna().all():
        return s
    cap = np.nanpercentile(s.values.astype("float64"), hi)
    return s.clip(upper=cap)


# ------------------ Main API ------------------ #
def build_features_nb(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features numériques à partir d'un DataFrame brut Home Credit.

    Paramètres
    ----------
    df_raw : pd.DataFrame
        Données brutes (application_train/test).

    Retour
    ------
    pd.DataFrame
        Features numériques (NaN tolérés pour imputation ultérieure).
    """
    X = df_raw.copy()

    # 1) Colonnes de base -> numériques
    for col in ("AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in ("DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 2) Anomalie DAYS_EMPLOYED == 365243
    if "DAYS_EMPLOYED" in X.columns:
        anom = X["DAYS_EMPLOYED"] == 365243
        X["DAYS_EMPLOYED_ANOM"] = anom.astype(int)
        X.loc[anom, "DAYS_EMPLOYED"] = np.nan

    # 3) EXT sources (mean + sum/3)
    X["ext_source_mean"] = _mean_if_exists(X, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    X["ext_source_sum"] = _sum_if_exists(X, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]) / 3.0

    # 4) Ratios financiers
    X["credit_income_ratio"] = _clip_percentile(_safe_ratio(X.get("AMT_CREDIT", np.nan), X.get("AMT_INCOME_TOTAL", np.nan)))
    X["annuity_income_ratio"] = _safe_ratio(X.get("AMT_ANNUITY", np.nan), X.get("AMT_INCOME_TOTAL", np.nan))
    X["credit_term_ratio"] = _safe_ratio(X.get("AMT_CREDIT", np.nan), X.get("AMT_ANNUITY", np.nan))

    if "AMT_GOODS_PRICE" in X.columns:
        X["goods_credit_ratio"] = _safe_ratio(X["AMT_GOODS_PRICE"], X.get("AMT_CREDIT", np.nan))
        X["goods_income_ratio"] = _safe_ratio(X["AMT_GOODS_PRICE"], X.get("AMT_INCOME_TOTAL", np.nan))

    # 5) Âge/ancienneté (positifs) + ratio d'ancienneté
    age_days = _abs_series(X, "DAYS_BIRTH")
    emp_days = _abs_series(X, "DAYS_EMPLOYED")
    X["days_employed_percent"] = _safe_ratio(emp_days, age_days)

    # 6) Flags propriétaires / documents si présents
    if "FLAG_OWN_CAR" in X.columns:
        X["FLAG_OWN_CAR"] = pd.to_numeric(X["FLAG_OWN_CAR"].map({"Y": 1, "N": 0}).fillna(X["FLAG_OWN_CAR"]), errors="coerce")
    if "FLAG_OWN_REALTY" in X.columns:
        X["FLAG_OWN_REALTY"] = pd.to_numeric(X["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0}).fillna(X["FLAG_OWN_REALTY"]), errors="coerce")

    flag_doc_cols = [c for c in X.columns if c.startswith("FLAG_DOCUMENT_")]
    X["flag_documents_sum"] = X[flag_doc_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if flag_doc_cols else np.nan

    own_cols = [c for c in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY") if c in X.columns]
    X["owner_sum"] = X[own_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if own_cols else np.nan

    if "OWN_CAR_AGE" in X.columns:
        X["carAge_income_ratio"] = _safe_ratio(pd.to_numeric(X["OWN_CAR_AGE"], errors="coerce"), X.get("AMT_INCOME_TOTAL", np.nan))

    # 7) "Avg_sum" : somme des *_AVG si dispo
    avg_cols = [c for c in X.columns if c.endswith("_AVG")]
    X["avg_sum"] = X[avg_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if avg_cols else np.nan

    # 8) "Stability" : normalisation simple de quelques DAYS_* (échelle 0..1 par colonne), puis somme
    stab_cols = ["DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "DAYS_EMPLOYED"]
    stab_parts = []
    for c in stab_cols:
        if c in X.columns:
            v = _abs_series(X, c)
            vmax = np.nanmax(v.values.astype("float64")) if v.notna().any() else np.nan
            part = v / vmax if vmax and np.isfinite(vmax) and vmax > 0 else pd.Series(np.nan, index=X.index)
        else:
            part = pd.Series(np.nan, index=X.index)
        stab_parts.append(part)
    X["stability"] = pd.concat(stab_parts, axis=1).sum(axis=1, min_count=1)

    # 9) Retour strictement numérique (on laisse l'imputation dans le pipeline modèle)
    X_num = X.select_dtypes(include=[np.number]).copy()
    return X_num
