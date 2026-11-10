"""Prépare gpu_bundle.zip pour le notebook notebooks/gpu_colab.ipynb (Colab).

Le zip contient le code (src/), les données brutes et les membres de la bibliothèque utilisés
en entrée des modèles de fondation (niveau 2). Sur Colab, le notebook exécute alors exactement
les mêmes commandes qu'en local (python -m src.models.tfm ...), avec le même découpage D/T et
les mêmes blocs.

Usage : python -m src.gpu_bundle            (après les étapes search, library et autogluon)
"""

import zipfile
from pathlib import Path

from src.models.library import LIB

BASES = ["lgbmgr_bag15", "catgr_1", "ag_LightGBMXT_BAG_L1"]


def main(out="gpu_bundle.zip"):
    files = [p for p in Path("src").rglob("*.py")] + [Path("pyproject.toml"),
             Path("data/raw/train.csv"), Path("data/raw/test.csv")]
    for m in BASES:
        files += [LIB / f"{m}.npz", LIB / f"{m}.json"]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"fichiers manquants (lancer d'abord les étapes précédentes) : {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.as_posix())
    print(f"{out} : {len(files)} fichiers, {Path(out).stat().st_size / 1e6:.0f} Mo")


if __name__ == "__main__":
    main()
