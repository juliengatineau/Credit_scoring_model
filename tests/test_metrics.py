import numpy as np
from credit_scoring.metrics import (
    confusion_counts,
    business_cost,
    business_score,
    find_optimal_threshold,
    eval_at_threshold,
)


def test_confusion_and_cost_simple():
    # y_true: 4 échantillons (2 pos, 2 neg)
    y_true = np.array([1, 0, 1, 0])
    # prédictions: 2 bons, 2 erreurs
    y_pred = np.array([1, 0, 0, 1])  # TP=1, TN=1, FP=1, FN=1
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    assert (tp, tn, fp, fn) == (1, 1, 1, 1)

    # coût = 10*FN + 1*FP = 10*1 + 1*1 = 11
    assert business_cost(y_true, y_pred, fn_cost=10.0, fp_cost=1.0) == 11.0

    # worst_cost = max( fn_cost*P=10*2=20, fp_cost*N=1*2=2 ) = 20
    # score = 1 - 11/20 = 0.45
    assert np.isclose(business_score(y_true, y_pred, 10.0, 1.0), 1.0 - 11.0 / 20.0)


def test_find_optimal_threshold_minimizes_cost_on_synthetic():
    # Probas synthétiques séparables ; meilleur seuil attendu vers 0.5
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    thr = find_optimal_threshold(y_true, y_proba, fn_cost=10.0, fp_cost=1.0, grid_size=501)
    assert 0.45 <= thr <= 0.55  # doit tomber près de la séparation


def test_eval_at_threshold_computes_all_metrics():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    # proba corrélées au label (bruitées)
    y_proba = np.clip(y_true * 0.7 + rng.normal(0.0, 0.2, size=200), 0, 1)

    thr = find_optimal_threshold(y_true, y_proba, fn_cost=5.0, fp_cost=1.0, grid_size=301)
    res = eval_at_threshold(y_true, y_proba, thr, fn_cost=5.0, fp_cost=1.0)

    assert 0.0 <= res.threshold <= 1.0
    assert 0.0 <= res.balanced_accuracy <= 1.0
    assert res.cost >= 0.0
    assert 0.0 <= res.business_score <= 1.0
