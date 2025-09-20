import sys
from pathlib import Path

print("===== Application Startup =====", file=sys.stderr)

# Log de versions (optionnel)
def _log_versions():
    try:
        import gradio, gradio_client
        print(
            f"gradio={gradio.__version__} gradio_client={gradio_client.__version__}",
            file=sys.stderr,
        )
        try:
            import websockets
            print(f"websockets={websockets.__version__}", file=sys.stderr)
        except Exception:
            print("websockets: not installed?", file=sys.stderr)
        try:
            import shap
            print(f"shap={shap.__version__}", file=sys.stderr)
        except Exception:
            print("shap: not installed?", file=sys.stderr)
    except Exception as e:
        print(f"Version log error: {e}", file=sys.stderr)

_log_versions()

# --- Shim "websockets.asyncio" (utile avec gradio_client==1.3.0 + websockets==12.x) ---
try:
    import websockets.asyncio.client  # noqa
except Exception:
    try:
        import types, websockets, websockets.client as _legacy_client
        mod = types.ModuleType("websockets.asyncio")
        mod.client = _legacy_client
        sys.modules["websockets.asyncio"] = mod
        sys.modules["websockets.asyncio.client"] = _legacy_client
        print("Shim websockets.asyncio -> websockets.client appliqué", file=sys.stderr)
    except Exception as e:
        print(f"Shim websockets.asyncio failed: {e}", file=sys.stderr)
# -------------------------------------------------------------------

# 🔑 AJOUTE le dossier src/ dans le PYTHONPATH
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
    print(f"Added to sys.path: {SRC}", file=sys.stderr)
else:
    print(f"WARNING: src directory not found at {SRC}", file=sys.stderr)

# Importe et construit l'UI
from credit_scoring.app.gradio_app import build_ui  # ton package sous src/credit_scoring/...

demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)