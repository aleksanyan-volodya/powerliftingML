"""Pipeline complet de la partie régression, étape par étape.

  search      recherche d'hyperparamètres (Optuna, validation croisée sur D)
  library     bibliothèque de membres : prédictions hors bloc sur D, sur T et sur le test
  autogluon   AutoML sur CPU, sans réseaux de neurones (environnement Python 3.12, voir README)
  gpu         étapes GPU : AutoGluon « extreme » (réseaux de neurones et modèles de fondation),
              TabICL et LimiX-2. En local s'il y a un GPU, sinon via notebooks/gpu_colab.ipynb
  ensemble    assemblage final, évaluation sur le test T et soumissions

Usage : python -m src.pipeline                       (toutes les étapes)
        python -m src.pipeline --steps ensemble      (une ou plusieurs étapes)
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASELINES = ["linear", "poly", "knn", "svr", "cart", "rf", "lgbm"]
BASES = ["lgbmgr_bag15", "catgr_1", "ag_LightGBMXT_BAG_L1"]
# correcteur enrichi : les mêmes bases, plus trois modèles de l'AutoGluon GPU
BASES_X = BASES + ["agx_CatBoost_BAG_L1", "agx_PrepLightGBM_BAG_L1", "agx_TabDPT-Turbo_BAG_L1"]
STEPS = ["search", "library", "autogluon", "gpu", "ensemble"]


def run(*cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    p.add_argument("--autogluon-python", default=str(Path(".venv-autogluon/Scripts/python.exe")
                                                     if sys.platform == "win32"
                                                     else Path(".venv-autogluon/bin/python")))
    p.add_argument("--autogluon-time", type=int, default=5400, help="secondes (CPU)")
    p.add_argument("--autogluon-gpu-time", type=int, default=14400, help="secondes (GPU)")
    args = p.parse_args()
    py = sys.executable

    if "search" in args.steps:
        run(py, "-m", "src.models.train", *BASELINES, "--timeout", "2400")
        run(py, "-m", "src.models.train", "lgbmgr", "--trials", "40", "--timeout", "4000")
        run(py, "-m", "src.models.train", "catgr", "--trials", "8", "--timeout", "5400")
    if "library" in args.steps:
        run(py, "-m", "src.models.library", *BASELINES, "--top-k", "3")
        run(py, "-m", "src.models.library", "lgbmgr", "--top-k", "3", "--bag-seeds", "15")
        run(py, "-m", "src.models.library", "catgr", "--top-k", "1")
    if "autogluon" in args.steps:
        run(args.autogluon_python, "-m", "src.models.autogluon_run",
            "--time-limit", str(args.autogluon_time))
    if "gpu" in args.steps:
        import torch
        if torch.cuda.is_available():
            run(args.autogluon_python, "-m", "src.models.autogluon_run", "--preset", "extreme",
                "--allow-nn", "--prefix", "agx_", "--time-limit", str(args.autogluon_gpu_time),
                "--path", "outputs/regression/autogluon_gpu/model")
            run(py, "-m", "src.models.tfm", "--model", "tabicl", "--name", "tabicl")
            run(py, "-m", "src.models.tfm", "--model", "tabicl", "--name", "tabicl_l2", "--base", *BASES)
            run(py, "-m", "src.models.tfm", "--model", "tabicl", "--name", "tabicl_l2x", "--base", *BASES_X)
            run(py, "-m", "src.models.tfm", "--model", "limix", "--name", "limix_l2", "--base", *BASES,
                "--budget-h", "3")
        else:
            run(py, "-m", "src.gpu_bundle")
            print("Pas de GPU : exécuter notebooks/gpu_colab.ipynb sur Colab avec gpu_bundle.zip, "
                  "décompresser gpu_results.zip à la racine du dépôt, puis lancer : "
                  "python -m src.pipeline --steps ensemble")
            return
    if "ensemble" in args.steps:
        run(py, "-m", "src.models.ensemble")


if __name__ == "__main__":
    main()
