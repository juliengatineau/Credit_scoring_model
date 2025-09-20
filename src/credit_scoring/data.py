# src/credit_scoring/data.py
"""
Data utilities for the credit scoring project.

Features
--------
- Download Home Credit Default Risk data from Kaggle (competition).
- Or fetch from direct URLs (train/test).
- Or generate a small synthetic demo dataset (no network).
- Support a custom Kaggle credentials file via --kaggle-json.

CLI
---
poetry run python -m credit_scoring.data --dest data --source kaggle --kaggle-json ~/ia/kaggle_api.json
poetry run python -m credit_scoring.data --dest data --source url --train-url <URL> --test-url <URL>
poetry run python -m credit_scoring.data --dest data --source demo
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Optional dependency for HTTP downloads
try:
    import requests  # type: ignore

    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    _HAS_REQUESTS = False


HOME_CREDIT_COMPETITION = "home-credit-default-risk"

# Minimal set (enough for training + demo app)
REQUIRED_FILES: tuple[str, ...] = (
    "application_train.csv",
    "application_test.csv",
)

# Full set (all competition tables)
FULL_FILES: tuple[str, ...] = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "POS_CASH_balance.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
)


# ------------------ Helpers ------------------ #
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_kaggle_creds_from_json(path: Path) -> None:
    """
    Load {username,key} from a JSON file and export KAGGLE_USERNAME/KAGGLE_KEY env vars.
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    os.environ["KAGGLE_USERNAME"] = str(data["username"])
    os.environ["KAGGLE_KEY"] = str(data["key"])


def _has_kaggle_creds(custom_path: Optional[Path] = None) -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if custom_path and custom_path.exists():
        return True
    # les deux emplacements que le client utilise selon les versions
    default1 = Path.home() / ".kaggle" / "kaggle.json"
    default2 = Path.home() / ".config" / "kaggle" / "kaggle.json"
    return default1.exists() or default2.exists()



def _download_file(url: str, dest_path: Path) -> None:
    if not _HAS_REQUESTS:
        raise RuntimeError("The 'requests' package is required for URL downloads. Install with: poetry add requests")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    chunk = 64 * 1024
    written = 0
    with open(dest_path, "wb") as f:
        for part in resp.iter_content(chunk_size=chunk):
            if part:
                f.write(part)
                written += len(part)
    if total and written < total * 0.95:
        raise IOError(f"Incomplete download ({written}/{total} bytes) from {url}")


# ------------------ Kaggle ------------------ #
def download_homecredit_kaggle(
    dest_dir: Path,
    files: Iterable[str],
    force: bool = False,
    kaggle_json: Optional[Path] = None,
) -> list[Path]:
    """
    Download specified files from the Kaggle competition and extract zips.
    """
    # ⚠️ CHARGER LES CREDS AVANT L'IMPORT (le package s'authentifie à l'import)
    if kaggle_json and kaggle_json.exists():
        _load_kaggle_creds_from_json(kaggle_json)

    # Maintenant seulement on importe
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as e:  # paquet manquant
        raise RuntimeError("The 'kaggle' package is required. Install with: poetry add kaggle") from e

    if not _has_kaggle_creds(kaggle_json):
        raise RuntimeError(
            "Kaggle credentials not found. Provide env vars, ~/.kaggle/kaggle.json or ~/.config/kaggle/kaggle.json, "
            "or pass --kaggle-json <path>."
        )

    _ensure_dir(dest_dir)
    api = KaggleApi()
    api.authenticate()

    out_paths: list[Path] = []
    for fname in files:
        csv_path = dest_dir / fname
        if csv_path.exists() and not force:
            out_paths.append(csv_path)
            continue

        zip_path = dest_dir / f"{fname}.zip"
        if zip_path.exists():
            zip_path.unlink()

        api.competition_download_file(
            HOME_CREDIT_COMPETITION,
            fname,
            path=str(dest_dir),
            force=True,
            quiet=False,
        )

        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest_dir)
            zip_path.unlink()

        if not csv_path.exists():
            for zf in dest_dir.glob("*.zip"):
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dest_dir)
                zf.unlink()

        if not csv_path.exists():
            raise FileNotFoundError(f"Failed to locate {fname} after Kaggle download.")
        out_paths.append(csv_path)

    return out_paths



# ------------------ URLs ------------------ #
def download_homecredit_from_urls(
    dest_dir: Path,
    train_url: str,
    test_url: Optional[str] = None,
    force: bool = False,
) -> list[Path]:
    """
    Download from provided direct URLs. Only supports train/test.
    """
    _ensure_dir(dest_dir)
    out: list[Path] = []

    mapping: list[tuple[str, str]] = [("application_train.csv", train_url)]
    if test_url:
        mapping.append(("application_test.csv", test_url))

    for fname, url in mapping:
        path = dest_dir / fname
        if path.exists() and not force:
            out.append(path)
            continue
        _download_file(url, path)
        if not path.exists():
            raise FileNotFoundError(f"Download failed for {fname} from {url}")
        out.append(path)
    return out


# ------------------ Demo generator ------------------ #
def generate_homecredit_demo(dest_dir: Path, n: int = 1000, seed: int = 123, include_test: bool = True) -> list[Path]:
    """
    Generate a synthetic application_train.csv (and application_test.csv) to test the pipeline offline.
    """
    _ensure_dir(dest_dir)
    rng = np.random.default_rng(seed)

    income = rng.uniform(80_000, 200_000, size=n)
    credit = rng.uniform(50_000, 400_000, size=n)
    annuity = credit / rng.uniform(8, 20, size=n)
    ext1 = rng.beta(5, 2, size=n)
    ext2 = rng.beta(4, 3, size=n)
    ext3 = rng.beta(6, 2, size=n)
    days_birth = -rng.integers(25, 65, size=n) * 365
    days_employed = -rng.integers(0, 20, size=n) * 365
    idx = rng.choice(n, size=max(1, n // 20), replace=False)
    days_employed[idx] = 365243  # anomaly marker used by Home Credit

    df = pd.DataFrame(
        {
            "AMT_CREDIT": credit,
            "AMT_INCOME_TOTAL": income,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
        }
    )

    ratio = credit / np.maximum(income, 1.0)
    risk = (1 - (ext1 + ext2 + ext3) / 3) * 0.7 + np.clip(ratio, 0, 5) * 0.3
    p = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)
    y = (p > np.quantile(p, 0.7)).astype(int)

    train_path = dest_dir / "application_train.csv"
    df_train = df.copy()
    df_train["TARGET"] = y
    df_train.to_csv(train_path, index=False)

    out = [train_path]
    if include_test:
        test_path = dest_dir / "application_test.csv"
        df.to_csv(test_path, index=False)
        out.append(test_path)
    return out


# ------------------ Orchestrator ------------------ #
@dataclass
class DataConfig:
    dest: Path
    source: str = "auto"  # "auto" | "kaggle" | "url" | "demo"
    train_url: Optional[str] = None
    test_url: Optional[str] = None
    force: bool = False
    full: bool = False
    kaggle_json: Optional[Path] = None


def ensure_homecredit_data(cfg: DataConfig) -> list[Path]:
    """
    Ensure the target files exist in cfg.dest according to the selected strategy.

    Strategy:
      - auto   : kaggle if creds are present; else URL if given; else demo.
      - kaggle : Kaggle (requires creds) or explicit error.
      - url    : direct URLs (train_url required).
      - demo   : synthetic dataset.

    Returns
    -------
    list[Path]: paths to available files.
    """
    _ensure_dir(cfg.dest)
    target_list = FULL_FILES if cfg.full else REQUIRED_FILES

    existing = [cfg.dest / f for f in target_list if (cfg.dest / f).exists()]
    if len(existing) == len(target_list) and not cfg.force:
        return existing

    source = cfg.source.lower()
    if source == "auto":
        if _has_kaggle_creds(cfg.kaggle_json):
            return download_homecredit_kaggle(cfg.dest, target_list, force=cfg.force, kaggle_json=cfg.kaggle_json)
        if cfg.train_url:
            return download_homecredit_from_urls(cfg.dest, cfg.train_url, cfg.test_url, force=cfg.force)
        return generate_homecredit_demo(cfg.dest)

    if source == "kaggle":
        return download_homecredit_kaggle(cfg.dest, target_list, force=cfg.force, kaggle_json=cfg.kaggle_json)
    if source == "url":
        if not cfg.train_url:
            raise ValueError("In 'url' mode, --train-url is required.")
        return download_homecredit_from_urls(cfg.dest, cfg.train_url, cfg.test_url, force=cfg.force)
    if source == "demo":
        return generate_homecredit_demo(cfg.dest)

    raise ValueError(f"Unknown source: {cfg.source!r} (expected: auto|kaggle|url|demo)")


# ------------------ CLI ------------------ #
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch Home Credit dataset (Kaggle / URL / demo).")
    p.add_argument("--dest", type=Path, default=Path("data"), help="Destination directory.")
    p.add_argument("--source", choices=["auto", "kaggle", "url", "demo"], default="auto", help="Fetch strategy.")
    p.add_argument("--train-url", type=str, default=None, help="Direct URL for application_train.csv (if source=url).")
    p.add_argument("--test-url", type=str, default=None, help="Direct URL for application_test.csv (if source=url).")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    p.add_argument("--full", action="store_true", help="Download all competition tables (not only application_*).")
    p.add_argument(
        "--kaggle-json",
        type=Path,
        default=None,
        help="Custom kaggle.json path (e.g., ~/ia/kaggle_api.json). Will set KAGGLE_USERNAME/KAGGLE_KEY env vars.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = DataConfig(
        dest=args.dest,
        source=args.source,
        train_url=args.train_url,
        test_url=args.test_url,
        force=args.force,
        full=args.full,
        kaggle_json=args.kaggle_json,
    )
    paths = ensure_homecredit_data(cfg)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
