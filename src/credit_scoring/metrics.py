"""
Métriques pour scoring crédit, avec focalisation "coût métier" et choix de seuil.

Principes retenus
-----------------
- On ne se contente pas de ROC AUC : on **optimise le seuil** pour le coût métier.
- Coût total = fn_cost * FN + fp_cost * FP.
- Score métier normalisé dans [0, 1] = 1 - cost / worst_cost,
  où worst_cost = max(fn_cost * P, fp_cost * N) (pire des stratégies triviales: tout refuser ou tout accepter).
  => Ce normaliseur est stable et interprétable. (Ton dénominateur initial était bancal.)

Fonctions principales
---------------------
- confusion_counts(y_true, y_pred) -> (TP, TN, FP, FN)
- business_cost(y_true, y_pred, fn_cost, fp_cost) -> float
- business_score(y_true, y_pred, fn_cost, fp_cost) -> float in [0, 1]
- find_optimal_threshold(y_true, y_proba, fn_cost, fp_cost, ...) -> seuil in [0, 1]
- eval_at_threshold(y_true, y_proba, thr, fn_cost, fp_cost) -> dict métriques
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """
    Calcule TP, TN, FP, FN (en supposant la classe positive = 1).

    Returns
    -------
    (tp, tn, fp, fn)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def business_cost(y_true: np.ndarray, y_pred: np.ndarray, fn_cost: float = 10.0, fp_cost: float = 1.0) -> float:
    """
    Coût total = fn_cost * FN + fp_cost * FP.
    """
    _, _, fp, fn = confusion_counts(y_true, y_pred)
    return fn_cost * fn + fp_cost * fp


def business_score(y_true: np.ndarray, y_pred: np.ndarray, fn_cost: float = 10.0, fp_cost: float = 1.0) -> float:
    """
    Score normalisé dans [0, 1] : 1 - cost / worst_cost.

    - worst_cost = max( fn_cost * P, fp_cost * N ) :
      * prédire tout 0 => coût = fn_cost * P (tous les 1 deviennent FN)
      * prédire tout 1 => coût = fp_cost * N (tous les 0 deviennent FP)
      On prend le pire des deux comme borne supérieure raisonnable du coût.

    Note : 1 = parfait (coût 0), 0 = aussi mauvais que la pire stratégie triviale.
    """
    y_true = np.asarray(y_true).astype(int)
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    worst = max(fn_cost * P, fp_cost * N) if (P + N) > 0 else 1.0
    cost = business_cost(y_true, y_pred, fn_cost, fp_cost)
    return float(1.0 - (cost / worst if worst > 0 else 1.0))


def _pred_from_proba(y_proba: np.ndarray, thr: float) -> np.ndarray:
    """Binarise les probabilités avec le seuil `thr`."""
    return (np.asarray(y_proba, dtype=float) >= float(thr)).astype(int)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    grid_size: int = 1001,
) -> float:
    """
    Cherche le seuil qui minimise le coût métier sur un maillage uniforme [0, 1].
    S'il existe **plusieurs** seuils minimaux, retourne le **milieu** de l'intervalle optimal.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_proba = np.clip(y_proba, 0.0, 1.0)

    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    costs = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        _, _, fp, fn = confusion_counts(y_true, y_pred)
        costs.append(fn_cost * fn + fp_cost * fp)
    costs = np.asarray(costs, dtype=float)

    min_cost = costs.min()
    idx = np.where(costs == min_cost)[0]
    # Milieu de l'intervalle [thr_min_opt, thr_max_opt]
    thr_opt = float((thresholds[idx[0]] + thresholds[idx[-1]]) / 2.0)
    return thr_opt



@dataclass
class EvalResult:
    threshold: float
    auc: float
    balanced_accuracy: float
    cost: float
    business_score: float


def eval_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> EvalResult:
    """
    Évalue des métriques clés à un seuil donné.

    Retourne
    --------
    EvalResult(threshold, auc, balanced_accuracy, cost, business_score)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = _pred_from_proba(y_proba, threshold)

    auc = float(roc_auc_score(y_true, y_proba)) if (len(np.unique(y_true)) == 2) else float("nan")
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    cost = float(business_cost(y_true, y_pred, fn_cost, fp_cost))
    bscore = float(business_score(y_true, y_pred, fn_cost, fp_cost))
    return EvalResult(threshold, auc, bacc, cost, bscore)

# --- métriques agrégées pour le train ---
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> dict:
    """
    Calcule un set de métriques standard + métier, cohérent avec train.py.

    Retourne un dict JSON-serializable contenant:
      - threshold, auc, balanced_accuracy, precision, recall, f1
      - tp, tn, fp, fn
      - cost (fn_cost*FN + fp_cost*FP), business_score (normalisé [0,1])
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    auc = float(roc_auc_score(y_true, y_proba)) if (len(np.unique(y_true)) == 2) else float("nan")
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    cost = float(business_cost(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost))
    bscore = float(business_score(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost))

    return {
        "threshold": float(threshold),
        "auc": auc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "cost": cost,
        "business_score": bscore,
    }
