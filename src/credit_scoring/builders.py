# src/credit_scoring/builders.py
from __future__ import annotations

"""
Sélectionneur de "feature builders" pour le projet.

- "simple": pipeline minimal existant (credit_scoring.features.build_features)
- "nb":     pipeline factorisé depuis tes notebooks (credit_scoring.nb_features.build_features_nb)

Usage:
    from credit_scoring.builders import get_builder
    build_fn = get_builder("nb")    # ou "simple"
    X = build_fn(df_raw)
"""

from importlib import import_module
from typing import Callable

# builder "simple" (déjà présent dans ton projet)
from credit_scoring.features import build_features as _simple


def _load_nb_builder() -> Callable:
    """
    Import paresseux du builder 'nb' pour éviter les erreurs si le fichier n'existe pas encore.
    Retourne la fonction build_features_nb du module credit_scoring.nb_features.
    """
    mod = import_module("credit_scoring.nb_features")
    fn = getattr(mod, "build_features_nb", None)
    if fn is None:
        raise AttributeError("Le module 'credit_scoring.nb_features' n'expose pas 'build_features_nb'.")
    return fn


def get_builder(name: str | None) -> Callable:
    """
    Retourne la fonction de construction des features selon `name`.

    Paramètres
    ----------
    name : {"simple", "nb"} ou None
        - "simple" : builder minimal existant
        - "nb"     : builder factorisé depuis les notebooks

    Retour
    ------
    Callable[[pd.DataFrame], pd.DataFrame]
        Fonction qui prend un DataFrame brut et retourne un DataFrame de features numériques.
    """
    key = (name or "simple").lower().strip()
    if key in ("simple", "default"):
        return _simple
    if key in ("nb", "notebook"):
        return _load_nb_builder()
    raise ValueError(f"Unknown feature builder: {name!r} (attendu: 'simple' ou 'nb').")
