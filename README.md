# Credit Scoring — Gradio Demo with SHAP (Portfolio)

> Binary credit risk scoring with **scikit‑learn pipelines**, **cost‑aware thresholding**, and **explainability** (local **SHAP waterfall** and **global feature importance**).  
> Includes a training CLI (**logistic / random forest / XGBoost / LightGBM**) and a **Gradio app** with model switcher.

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10–3.12-blue)](#requirements)
[![scikit‑learn](https://img.shields.io/badge/scikit--learn-1.4+-f89939)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.43.0-FF4B4B?logo=gradio)](https://gradio.app/)
[![Live Demo – HF Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-black?logo=huggingface)](https://huggingface.co/spaces/juliengatineau/credit_scoring_demo)

> • Gradio: https://huggingface.co/spaces/juliengatineau/credit_scoring_demo

---

## TL;DR

-   **Train** a scoring model with: `poetry run train-scoring --algorithm logistic|rf|xgb|lgbm --name my_model`  
    → saves `models/my_model.joblib`, `artifacts/my_model_metrics.json`, `artifacts/my_model_global_shap.json`.
-   **Serve** an interactive **Gradio** app: `poetry run scoring-app`
    -   Tab **Utilisateur**: pick a random profile, view **probability & decision**, and a **SHAP waterfall** (local).
    -   Tab **Global**: **pre‑computed** global importance (mean |SHAP|).
    -   Tab **Métriques modèle**: values + short **explanations** for AUC, Balanced Accuracy, Business Score, Cost.
-   **Data loading** in the app is automatic: looks for `data/application_test.csv`, falls back to `data/application_train.csv`, else shows a tiny synthetic sample. Features are aligned to those used at train time.
-   The **threshold** used for the decision is **optimized for business cost** at training time and is adjustable in the UI.

---

## Model & Explainability

-   **Preprocessing**: robust industrial pipeline (imputation, binarization of `Y/N/Yes/No`, OHE for categoricals, scaling where needed).
-   **Algorithms**: `logistic`, `rf` (RandomForest), `xgb` (XGBoost), `lgbm` (LightGBM).
-   **Thresholding**: we search the probability **threshold** that **minimizes** total business **cost** on train:
    -   `cost = fn_cost * FN + fp_cost * FP`, with defaults `fn_cost=10`, `fp_cost=1`.
-   **Business Score** (normalized, ↑ is better):  
    `business_score = 1 - cost / worst_cost`, where `worst_cost = max(fn_cost * P, fp_cost * N)` (worst of “predict all 0” or “predict all 1”).
-   **Global SHAP**: computed **once during training** and saved to `artifacts/{name}_global_shap.json`.
-   **Local SHAP**: computed **on the fly** in the app (waterfall plot).

> Dataset: designed for **Home Credit Default Risk** style tables (not included). You can place your CSVs under `data/`.

---

## Project Layout

```
.
├─ src/
│  └─ credit_scoring/
│     ├─ app/
│     │  └─ gradio_app.py        # Gradio UI (tabs: Utilisateur / Global / Métriques)
│     ├─ preprocess.py           # industrial feature engineering & preprocess bundle
│     ├─ metrics.py              # AUC, balanced acc, business cost/score, optimal threshold search
│     ├─ train.py                # CLI training entrypoint
│     ├─ model.py, builders.py   # helpers (e.g., model factories, feature builders)
│     └─ nb_features.py          # optional notebook parity features
├─ app/
│  └─ app.py                     # minimal wrapper for HF Spaces (imports Gradio UI)
├─ tests/                        # unit tests (features, metrics, model, train)
├─ data/                         # (ignored) put application_train/test.csv here
├─ models/                       # (ignored) trained pipelines as .joblib
├─ artifacts/                    # (ignored) metrics JSON + global SHAP JSON
├─ pyproject.toml                # Poetry project (pins working versions)
├─ requirements.txt              # For Spaces (pinned, minimal)
├─ README.md
└─ LICENSE
```

> **Note:** `data/`, `models/`, and `artifacts/` are **git‑ignored** by default. Regenerate locally with the training CLI.

---

## Local Setup (Poetry)

```bash
# Python 3.10–3.12 recommended
poetry env use python3.12
poetry install --no-root

# (Optional) run tests
poetry run pytest -q
```

### Train models

```bash
# Logistic regression
poetry run train-scoring \
  --data data/application_train.csv \
  --target TARGET \
  --algorithm logistic \
  --sample-frac 0.30 \
  --random-state 42 \
  --name logistic_nb

# XGBoost (requires xgboost)
poetry run train-scoring \
  --data data/application_train.csv \
  --target TARGET \
  --algorithm xgb \
  --sample-frac 0.30 \
  --random-state 42 \
  --name xgb_nb

# LightGBM (requires lightgbm)
poetry run train-scoring \
  --data data/application_train.csv \
  --target TARGET \
  --algorithm lgbm \
  --sample-frac 0.30 \
  --random-state 42 \
  --name lgbm_nb
```

Artifacts appear under:

-   `models/{name}.joblib` — full **Pipeline(preprocess, clf)**
-   `artifacts/{name}_metrics.json` — metrics + **train threshold**
-   `artifacts/{name}_global_shap.json` — **global SHAP** importance (sorted in app)

### Launch the app

```bash
poetry run scoring-app
# → opens a local Gradio server and may print a share URL
```

---

## App Features

-   **Model selector** (dropdown): pick any trained model found via `artifacts/*_metrics.json`.
-   **Seuil de décision** (slider): adjust threshold used to binarize probabilities.
-   **Utilisateur** tab:
    -   **Score utilisateur** block: ID, probability, decision, and a compact inline metrics summary.
    -   **Données utilisateur** (1 row, horizontal).
    -   **SHAP local (waterfall)**: for the selected profile (feature names + values shown).
-   **Global** tab: horizontal **bar chart** of mean |SHAP| (most important **on top**).
-   **Métriques modèle** tab: detailed explanations for **AUC**, **Balanced Accuracy**, **Business Score**, and **Cost**.

---

## Requirements

-   **Python** 3.10–3.12
-   **Poetry** (recommended) or `pip`
-   Core deps: `pandas`, `numpy`, `scikit-learn`, `joblib`, `shap`, `xgboost` (opt), `lightgbm` (opt), `gradio`

The **working set** used in this repo (also used on HF Spaces) is pinned in `pyproject.toml`:

```toml
pandas = "^2.2"
numpy = "^1.26"
scikit-learn = "^1.4"
joblib = "^1.4"
shap = "^0.45"
xgboost = "^3.0.5"
lightgbm = "^4.3"
gradio = "4.43.0"
gradio-client = "1.3.0"
```

> These versions avoid a few incompatibility pitfalls (see Troubleshooting).

---

## Troubleshooting

-   **`ModuleNotFoundError: websockets.asyncio`** on Spaces  
    → Pin `gradio==4.43.0`, `gradio_client==1.3.0`, and `websockets>=10,<13` (e.g., 12.0). The small **shim** in `app/app.py` maps `websockets.asyncio` to `websockets.client`.
-   **Gradio client JSON schema bool crash** in older stacks  
    → We guard with a small hotfix inside the UI (`json_schema_to_python_type = lambda schema: "Any"`).
-   **Local SHAP figure cropped on the left**  
    → We set generous left margins; you can tweak `subplots_adjust(left=...)` or adjust plot CSS (`#user-shap img { height: 80vh; }`).
-   **Different raw CSV schemas**  
    → The app **aligns** runtime features to what the model expects; missing columns are filled neutral (0/NaN) through the pipeline.

---

## Acknowledgements

-   Inspired by the **Home Credit Default Risk** dataset (Kaggle).
-   Built with **scikit‑learn**, **LightGBM**, **XGBoost**, **SHAP**, and **Gradio**.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).
