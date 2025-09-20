# src/credit_scoring/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

_OHE_KW = {"handle_unknown": "ignore"}
_major, _minor = (int(x) for x in sklearn.__version__.split(".")[:2])

if (_major, _minor) >= (1, 2):
    _OHE_KW["sparse_output"] = False  # sklearn>=1.2
else:
    _OHE_KW["sparse"] = False         # sklearn<1.2

# --------- 1) Feature engineering (fidèle à ton notebook, mais encapsulé) ---------
class FeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Crée quelques features dérivées, de façon tolérante aux colonnes manquantes
    et aux booléens encodés en 'Y'/'N', 'Yes'/'No', True/False, etc.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _to_float(self, s: pd.Series) -> pd.Series:
        """Convertit souplement vers float (gère 'Y'/'N', True/False, 'Yes'/'No')."""
        if s is None:
            return pd.Series(dtype="float64")
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        # mappe les binaires usuels, puis force en numérique
        s = s.replace({"Y": 1, "N": 0, "Yes": 1, "No": 0, True: 1, False: 0})
        return pd.to_numeric(s, errors="coerce")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- Sécuriser quelques colonnes de base si présentes ---
        if "DAYS_BIRTH" in df.columns:
            # convertit en âge (années positives)
            df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).astype(float)

        if "DAYS_EMPLOYED" in df.columns:
            # 365243 = valeur sentinelle (anomalie). Flag + remplace par NaN
            anom_mask = df["DAYS_EMPLOYED"] == 365243
            df["DAYS_EMPLOYED_ANOM"] = anom_mask.astype(int)
            df.loc[anom_mask, "DAYS_EMPLOYED"] = np.nan
            df["EMP_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).astype(float)

        # --- Ratios financiers basiques (si colonnes présentes) ---
        if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["CREDIT_INCOME_PCT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["ANNUITY_INCOME_PCT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

        if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
            df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

        # --- Moyenne des EXT_SOURCE (tolère manquants) ---
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

        # --- owner_sum: gère 'Y'/'N' proprement ---
        if "FLAG_OWN_CAR" in df.columns or "FLAG_OWN_REALTY" in df.columns:
            car = self._to_float(df["FLAG_OWN_CAR"]) if "FLAG_OWN_CAR" in df.columns else 0.0
            realty = self._to_float(df["FLAG_OWN_REALTY"]) if "FLAG_OWN_REALTY" in df.columns else 0.0
            df["OWNER_SUM"] = (pd.Series(car).fillna(0) + pd.Series(realty).fillna(0)).astype(float)

        # --- Stabilité (si colonnes présentes) ---
        stab_parts = [c for c in ["DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "DAYS_EMPLOYED"] if c in df.columns]
        if stab_parts:
            # On prend la somme des durées en années (valeurs positives)
            tmp = df[stab_parts].copy()
            for c in stab_parts:
                tmp[c] = np.abs(tmp[c]) / 365.25
            df["STABILITY_SCORE"] = tmp.sum(axis=1)

        return df


# --------- 2) Construction du préprocesseur ---------
@dataclass
class PreprocessorBundle:
    pipeline: Pipeline
    feature_names: List[str]


def make_preprocessor(df_sample: pd.DataFrame) -> PreprocessorBundle:
    """
    Construit un préprocesseur:
      - FeatureEngineer()
      - ColumnTransformer(num: imputer+scaler, cat: OneHot)
    Détecte les colonnes num/obj sur un échantillon (typiquement le train complet).
    Retourne le pipeline prêt à .fit(), plus la liste des noms de features finale (une fois fit).
    """
    # 1) Appliquer FE pour déterminer les colonnes après FE
    fe = FeaturesEngineer()
    df_fe = fe.transform(df_sample)

    # 2) Détection colonnes
    cat_cols = df_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    # Ne pas inclure TARGET si présent
    if "TARGET" in num_cols:
        num_cols.remove("TARGET")
    if "TARGET" in cat_cols:
        cat_cols.remove("TARGET")

    # 3) Transfos
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**_OHE_KW)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("fe", fe),
        ("ct", ct),
    ])

    # 4) Fit sur l'échantillon pour exposer get_feature_names_out
    pipe.fit(df_sample)
    # Récup noms: num + cat(ohe)
    ohe: OneHotEncoder = pipe.named_steps["ct"].named_transformers_["cat"].named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    feature_names = num_cols + cat_names

    return PreprocessorBundle(pipeline=pipe, feature_names=feature_names)


def transform_to_frame(preprocessor: Pipeline, feature_names: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le préprocesseur et retourne un DataFrame avec les noms de colonnes alignés.
    """
    arr = preprocessor.transform(df)
    X = pd.DataFrame(arr, columns=feature_names, index=df.index)
    return X
