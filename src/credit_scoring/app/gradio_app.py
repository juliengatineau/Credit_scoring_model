from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# --- Hotfix gradio_client (1.3.0) : API schema -> bool crash ---
try:
    import gradio_client.utils as _gcu  # type: ignore
    _gcu.json_schema_to_python_type = lambda schema: "Any"
except Exception:
    pass
# ---------------------------------------------------------------

from credit_scoring.builders import get_builder
from credit_scoring.preprocess import transform_to_frame


# ======================== Artefacts dynamiques ========================

class ModelArtifacts:
    """Regroupe les artefacts chargés pour un modèle donné."""
    def __init__(self, name: str, model, metrics: Dict[str, Any], global_shap: Optional[Dict[str, Any]]):
        self.name = name
        self.model = model
        self.metrics = metrics
        self.global_shap = global_shap
        self.threshold: float = float(metrics.get("threshold", 0.5))
        self.feature_columns: List[str] = list(metrics.get("feature_names") or metrics.get("feature_columns") or [])
        self.builder_name: str = metrics.get("feature_builder") or "nb"
        self.build_fn = get_builder(self.builder_name)


def _list_available_models() -> List[str]:
    """Liste les modèles disponibles à partir des fichiers artifacts/{name}_metrics.json."""
    names = []
    for p in sorted(Path("artifacts").glob("*_metrics.json")):
        base = p.name.replace("_metrics.json", "")
        names.append(base)
    # Fallback legacy si aucun trouvé
    if not names and Path("artifacts/metrics.json").exists():
        names = ["default"]
    return names


def _load_artifacts_for(name: str) -> Tuple[Optional[ModelArtifacts], str]:
    """Charge model + metrics + global_shap pour un 'name' donné."""
    try:
        if name == "default":
            metrics_path = Path("artifacts/metrics.json")
            model_path = Path("models/model.joblib")
            global_path = Path("artifacts/default_global_shap.json")  # peut ne pas exister
        else:
            metrics_path = Path(f"artifacts/{name}_metrics.json")
            model_path = Path(f"models/{name}.joblib")
            global_path = Path(f"artifacts/{name}_global_shap.json")

        if not metrics_path.exists() or not model_path.exists():
            return None, f"❌ Artefacts manquants pour '{name}' (metrics/model)."

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        model = joblib.load(model_path)
        global_shap = None
        if global_path.exists():
            with global_path.open("r", encoding="utf-8") as f:
                global_shap = json.load(f)

        art = ModelArtifacts(name=name, model=model, metrics=metrics, global_shap=global_shap)
        notes = ""
        return art, notes
    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Erreur chargement artefacts '{name}': {e}"


# ======================== Données & features ========================

def _choose_default_csv() -> Optional[Path]:
    for p in [Path("data/application_test.csv"), Path("data/application_train.csv")]:
        if p.exists():
            return p
    return None


def _load_csv_auto() -> Tuple[pd.DataFrame, str]:
    p = _choose_default_csv()
    if p is None:
        demo = pd.DataFrame({
            "AMT_CREDIT": [200000, 500000],
            "AMT_INCOME_TOTAL": [180000, 120000],
            "AMT_ANNUITY": [18000, 45000],
            "DAYS_BIRTH": [-44 * 365, -31 * 365],
            "DAYS_EMPLOYED": [-10 * 365, 365243],
            "EXT_SOURCE_1": [0.55, 0.12],
            "EXT_SOURCE_2": [0.62, 0.18],
            "EXT_SOURCE_3": [0.57, 0.09],
        })
        return demo, "Exemple synthétique (aucun fichier data/ trouvé)."
    df = pd.read_csv(p)
    return df, ""


def _align_features(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[[c for c in cols if c in X.columns]]


# ======================== SHAP (local uniquement dans l’app) ========================

def _make_explainer_numeric(pipeline, df_raw, feature_names_hint: Optional[List[str]] = None):
    """
    Construit un explainer SHAP dans l'espace transformé par le préprocesseur du pipeline.
    Retourne: (explainer, X_num (np.ndarray), feature_names (List[str]))
    """
    import shap
    import numpy as np

    # Récupérer le step preprocess et l'estimateur final
    preproc = None
    final_est = pipeline
    if hasattr(pipeline, "named_steps"):
        steps = list(pipeline.named_steps.items())
        if steps:
            final_est = steps[-1][1]
        preproc = pipeline.named_steps.get("prep", None)

    # Transformer -> DataFrame nommée si possible
    X_df = None
    feat_names: Optional[List[str]] = None

    if preproc is not None and feature_names_hint:
        try:
            # meilleur chemin : on force les noms connus du train
            X_df = transform_to_frame(preproc, feature_names_hint, df_raw)
            feat_names = list(X_df.columns)
        except Exception:
            X_df = None
            feat_names = None

    if X_df is None and preproc is not None:
        try:
            X_num = preproc.transform(df_raw)
            try:
                # tente avec input_features pour de meilleurs noms
                feat_names = list(preproc.get_feature_names_out(input_features=df_raw.columns))
            except TypeError:
                feat_names = list(preproc.get_feature_names_out())
            except Exception:
                feat_names = None
            if feat_names is not None and len(feat_names) == getattr(X_num, "shape", (0, 0))[1]:
                import pandas as pd
                X_df = pd.DataFrame(X_num, columns=feat_names)
            else:
                X_df = None
        except Exception:
            X_df = None

    if X_df is None:
        # fallback : numérique brut
        import pandas as pd
        num_df = df_raw.select_dtypes(include=[np.number])
        if num_df.empty:
            # tout a échoué -> on convertit tout en numériques où possible
            num_df = df_raw.apply(pd.to_numeric, errors="coerce")
        X_df = num_df
        feat_names = list(X_df.columns)

    X_num = X_df.to_numpy()
    feat_names = list(feat_names)

    # Explainer
    bg = X_num[: min(200, X_num.shape[0])]
    est_name = final_est.__class__.__name__.lower()
    if any(k in est_name for k in ["xgb", "lgbm", "forest", "tree", "boost", "gbm"]):
        explainer = shap.TreeExplainer(final_est, bg, model_output="probability")
    elif any(k in est_name for k in ["logistic", "linear", "sgd"]):
        explainer = shap.LinearExplainer(final_est, bg)
    else:
        def predict_num(A):
            return final_est.predict_proba(A)[:, 1]
        explainer = shap.KernelExplainer(predict_num, bg)

    return explainer, X_num, feat_names





def _plot_shap_local(explainer, x_num_row, feat_names, max_display: int = 15):
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    exp = explainer(x_num_row)  # 1 ligne
    try:
        ex = exp[0]
    except Exception:
        ex = exp

    # forcer noms + valeurs pour les libellés "feat = value"
    ex.feature_names = list(feat_names)
    if getattr(ex, "data", None) is None:
        ex.data = np.asarray(x_num_row)[0]

    # figure plus large
    fig = plt.figure()
    shap.plots.waterfall(ex, max_display=max_display, show=False)
    plt.gcf().set_size_inches(12,12)

    # marge gauche plus grande pour éviter toute coupure
    plt.gcf().subplots_adjust(left=0.5, right=0.98, top=0.92, bottom=0.12)
    return fig







def _plot_global_from_json(payload: Dict[str, Any]):
    """Barh trié décroissant (plus important en haut) à partir du JSON global SHAP.
    Accepte deux formats :
      A) {"features":[...], "mean_abs_shap":[...]}
      B) {"feat1": val1, "feat2": val2, ...}
    """
    import numpy as np
    import matplotlib.pyplot as plt

    feats: List[str]; vals: List[float]

    if isinstance(payload, dict) and "features" in payload and "mean_abs_shap" in payload:
        feats = list(map(str, payload.get("features") or []))
        vals  = list(map(float, payload.get("mean_abs_shap") or []))
    elif isinstance(payload, dict):
        # format mapping {feature: importance}
        items = list(payload.items())
        feats = [str(k) for k, _ in items]
        vals  = [float(v) for _, v in items]
    else:
        return None

    # Tri décroissant
    pairs = sorted(zip(feats, vals), key=lambda kv: kv[1], reverse=True)
    if not pairs:
        return None
    feats, vals = zip(*pairs)
    y = np.arange(len(feats))

    fig = plt.figure(figsize=(10.5, 6.4))
    ax = fig.add_subplot(111)
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Importance (|SHAP| moyen)")
    ax.set_title("Importance globale des variables")
    ax.invert_yaxis()  # plus important en haut

    max_len = max((len(x) for x in feats), default=10)
    left = min(0.08 + 0.012 * max_len, 0.55)
    fig.subplots_adjust(left=left, right=0.98, top=0.92, bottom=0.10)
    return fig


# ======================== Callbacks ========================

def _format_user_status(idx: int, thr: float, df_raw: pd.DataFrame, proba: np.ndarray, metrics: Dict[str, Any]) -> str:
    p = float(proba[idx])
    decision = int(p >= float(thr))

    ident = ""
    for key in ("SK_ID_CURR", "ID", "id"):
        if key in df_raw.columns:
            ident = f"{key}={df_raw.iloc[idx][key]}"
            break
    if not ident:
        ident = f"index={idx}"

    # Bloc unique : titre + score + métriques
    detail = f"""
<hr style="height:2px;border:none;background:#999;margin:10px 0;">
<h2 style="font-size:1.5rem;margin:0 0 12px 0;">Score utilisateur</h2>
<p style="font-size:1.15rem;margin:4px 0 0;">
  <strong>{ident}</strong> : proba défaut = <strong>{p:.3f}</strong> &rarr; décision = <strong>{decision}</strong>
</p>
<hr style="height:2px;border:none;background:#999;margin:10px 0;">
    """.strip()

    return detail



def _fmt_val(v, nd=4):
    try:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{nd}f}"
    except Exception:
        pass
    return "—"

def _metrics_markdown(m: Dict[str, Any]) -> str:
    """Construit un Markdown avec explications + valeurs pour chaque métrique."""
    if not m:
        return (
            "### Métriques du modèle\n\n"
            "_Aucune métrique disponible pour ce modèle._"
        )

    thr  = m.get("threshold")
    auc  = m.get("auc")
    bacc = m.get("balanced_accuracy")
    bscore = m.get("business_score") or m.get("business_metric")
    cost = m.get("cost") or m.get("business_cost")
    fnc  = m.get("fn_cost")
    fpc  = m.get("fp_cost")

    parts = [
        "### Métriques du modèle",
        "",
        "#### Seuil de décision (train)",
        "Seuil appliqué aux probabilités pour prendre la décision (1 si proba ≥ seuil, sinon 0). "
        "Il a été choisi pour **minimiser le coût métier** sur le jeu d’entraînement.\n",
        f"**Valeur :** `{_fmt_val(thr, nd=3)}`",
        "",
        "#### AUC (ROC AUC)",
        "Probabilité qu’un positif ait une proba supérieure à un négatif. **Indépendant du seuil**. "
        "0.5 ≈ aléatoire, 1.0 = parfait.\n",
        f"**Valeur :** `{_fmt_val(auc)}`",
        "",
        "#### Balanced accuracy",
        "Moyenne de la sensibilité (rappel positifs) et de la spécificité (rappel négatifs). "
        "Utile en cas de **déséquilibre de classes**.\n",
        f"**Valeur :** `{_fmt_val(bacc)}`",
        "",
        "#### Business score",
        "Score normalisé dans [0,1] : `1 - coût / coût_pire`, avec "
        "`coût = fn_cost * FN + fp_cost * FP` et `coût_pire = max(fn_cost * P, fp_cost * N)` "
        "(pire des stratégies *tout refuser* ou *tout accepter*).\n",
        f"**Valeur :** `{_fmt_val(bscore)}`",
        "",
        "#### Coût métier",
        "Coût total au seuil choisi : `fn_cost * FN + fp_cost * FP`.\n",
        f"**Valeur :** `{_fmt_val(cost)}`" + (
            f" (avec `fn_cost={_fmt_val(fnc, nd=2)}`, `fp_cost={_fmt_val(fpc, nd=2)}`)"
            if (fnc is not None and fpc is not None) else ""
        ),
    ]
    return "\n".join(parts)



def cb_init_model(name: str, thr: float):
    """Charge artefacts + dataset brut + proba + explainer local + plot global."""
    art, notes = _load_artifacts_for(name)
    if art is None:
        # must return 12 outputs
        metrics_md = _metrics_markdown({})
        return (
            notes or f"❌ Artefacts manquants pour '{name}'.",
            0.5,
            pd.DataFrame(),                 # st_raw
            np.empty((0, 0)),               # st_X (num matrix)
            np.array([]),                   # st_proba
            None,                           # st_explainer
            -1,                             # st_idx
            "—",                            # st_model_name
            None,                           # global_fig
            "—",                            # global_status
            {},                             # st_metrics
            metrics_md,                     # metrics_md
        )

    try:
        df_raw, msg = _load_csv_auto()

        # scorer directement le pipeline sur le brut
        proba = art.model.predict_proba(df_raw)[:, 1]

        # construire l’explainer dans l’espace transformé
        feat_hint = art.metrics.get("feature_names") or art.feature_columns
        explainer, X_num, feat_names = _make_explainer_numeric(
            art.model, df_raw, feature_names_hint=feat_hint
        )

        # idx initial
        rng = np.random.default_rng(0)
        idx = int(rng.integers(0, len(X_num))) if len(X_num) else -1

        # SHAP global (pré-calculé)
        if art.global_shap:
            global_fig = _plot_global_from_json(art.global_shap)
            global_status = (
                f"✅ Importance globale pré-calculée "
                f"(n={art.global_shap.get('n_sample','?')})."
            )
        else:
            global_fig = None
            global_status = "⚠️ SHAP global manquant pour ce modèle."

        init_status = f"{notes} {msg}".strip()
        model_metrics = art.metrics or {}
        metrics_md = _metrics_markdown(model_metrics)

        # SUCCESS: must return 12 outputs
        return (
            init_status,
            float(art.threshold),
            df_raw,                         # st_raw
            X_num,                          # st_X (num matrix)
            proba,                          # st_proba
            (explainer, feat_names),        # st_explainer
            idx,                            # st_idx
            name,                           # st_model_name
            global_fig,
            global_status,
            model_metrics,                  # st_metrics
            metrics_md,                     # metrics_md
        )

    except Exception as e:
        traceback.print_exc()
        # ERROR: must return 12 outputs
        return (
            f"❌ Erreur init modèle '{name}': {e}",
            float(art.threshold),
            pd.DataFrame(),                 # st_raw
            np.empty((0, 0)),               # st_X
            np.array([]),                   # st_proba
            None,                           # st_explainer
            -1,                             # st_idx
            name,                           # st_model_name
            None,                           # global_fig
            "—",                            # global_status
            {},                             # st_metrics
            _metrics_markdown({}),          # metrics_md
        )





def cb_pick_random(thr, df_raw, X_num, proba, explainer_pack, _idx, metrics):
    if X_num is None or len(X_num) == 0:
        return "❌ Aucune donnée.", pd.DataFrame(), None, -1
    explainer, feat_names = explainer_pack if isinstance(explainer_pack, tuple) else (None, None)

    rng = np.random.default_rng()
    idx = int(rng.integers(0, len(X_num)))
    detail = _format_user_status(idx, thr, df_raw, proba, metrics)

    fig = None
    if explainer is not None:
        try:
            x_row_num = X_num[idx:idx+1]
            fig = _plot_shap_local(explainer, x_row_num, feat_names, max_display=15)
        except Exception as e:
            detail += f"\n\n> ⚠️ SHAP local indisponible: `{e}`"

    return detail, df_raw.iloc[[idx]], fig, idx


def cb_refresh_user(thr, df_raw, X_num, proba, explainer_pack, idx, metrics):
    if idx < 0 or idx >= len(X_num):
        return "❌ Index invalide.", pd.DataFrame(), None
    detail = _format_user_status(int(idx), thr, df_raw, proba, metrics)

    fig = None
    if isinstance(explainer_pack, tuple):
        explainer, feat_names = explainer_pack
        try:
            x_row_num = X_num[int(idx):int(idx)+1]
            fig = _plot_shap_local(explainer, x_row_num, feat_names, max_display=15)
        except Exception as e:
            detail += f"\n\n> ⚠️ SHAP local indisponible: `{e}`"

    return detail, df_raw.iloc[[int(idx)]], fig




# ======================== UI ========================

def build_ui() -> gr.Blocks:
    CUSTOM_CSS = """
#user-shap img {
  max-height: 70vh;
  height: auto;
  width: auto;
}
"""

    with gr.Blocks(title="Credit Scoring — Home Credit", css=CUSTOM_CSS) as demo:
        gr.Markdown("# Credit Scoring — Démo")
        gr.Markdown(
            "Sélectionne un **modèle**. "
            "Le slider ajuste le **seuil**.  \n"
            "Onglet **Utilisateur** : un profil aléatoire, ses **données** (horizontal), son **score** et un **SHAP local**.  \n"
            "Onglet **Global** : **importance globale** des variables (pré-calculée à l'entraînement)."
        )

        # Modèles disponibles
        choices = _list_available_models()
        model_selector = gr.Dropdown(
            choices=choices, value=(choices[0] if choices else None), label="Modèle"
        )

        thr = gr.Slider(0.0, 1.0, value=0.5, step=0.001, label="Seuil de décision")

        # States
        st_raw = gr.State(pd.DataFrame())
        st_X = gr.State(np.empty((0, 0)))
        st_proba = gr.State(np.array([]))
        st_explainer = gr.State(None)
        st_idx = gr.State(-1)
        st_model_name = gr.State("—")
        st_metrics = gr.State({})


        # Init status
        init_status = gr.Markdown()

        with gr.Tabs():
            with gr.TabItem("Utilisateur"):
                btn_rand = gr.Button("Choisir un utilisateur aléatoire")
                user_detail = gr.Markdown()
                user_raw = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="Données utilisateur (1 ligne)")
                user_fig = gr.Plot(label="SHAP local (waterfall)", elem_id="user-shap",)

                btn_rand.click(
                    cb_pick_random,
                    inputs=[thr, st_raw, st_X, st_proba, st_explainer, st_idx, st_metrics],
                    outputs=[user_detail, user_raw, user_fig, st_idx],
                )
                thr.change(
                    cb_refresh_user,
                    inputs=[thr, st_raw, st_X, st_proba, st_explainer, st_idx, st_metrics],
                    outputs=[user_detail, user_raw, user_fig],
                )


            with gr.TabItem("Global"):
                global_fig = gr.Plot(label="Importance globale (|SHAP| moyen)")
                global_status = gr.Markdown()
            with gr.TabItem("Métriques modèle"):
                metrics_md = gr.Markdown("### Métriques du modèle\n\n_Chargement..._")

        # Charger le modèle sélectionné
        def _switch_model(name, _thr):
            return cb_init_model(name, _thr)

        model_selector.change(
            _switch_model, inputs=[model_selector, thr],
            outputs=[
                init_status, thr, st_raw, st_X, st_proba, st_explainer, st_idx,
                st_model_name, global_fig, global_status, st_metrics, metrics_md
            ],
        )

        if choices:
            demo.load(
                _switch_model, inputs=[model_selector, thr],
                outputs=[
                    init_status, thr, st_raw, st_X, st_proba, st_explainer, st_idx,
                    st_model_name, global_fig, global_status, st_metrics, metrics_md
                ],
            )


        else:
            init_status.value = "❌ Aucun modèle trouvé. Entraîne au moins un modèle avec `--name`."

    return demo


def main() -> None:
    ui = build_ui()
    ui.launch(share=True)


if __name__ == "__main__":
    main()
