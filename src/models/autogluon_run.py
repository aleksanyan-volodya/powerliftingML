"""AutoML avec AutoGluon, à lancer depuis la racine du dépôt avec l'environnement
Python 3.12 (voir README) :

    .venv-autogluon/Scripts/python.exe -m src.models.autogluon_run --time-limit 5400

AutoGluon combine trois ingrédients : un PORTFOLIO de bonnes configurations 
(preset best_quality : configurations « zero-shot » issues de TabRepo, au lieu d'une recherche
d'hyperparamètres), un BAGGING en K blocs de chaque modèle (chaque point de D reçoit une
prédiction hors bloc) et une sélection gloutonne (GES) des modèles, éventuellement empilés sur
un second niveau (stacking).

Deux configurations :
  - CPU (défaut) : preset best_quality, réseaux de neurones et modèles de fondation exclus ;
    membres ag_<modèle> ;
  - GPU (--preset extreme --allow-nn --prefix agx_) : portfolio d'AutoGluon 1.5 « extreme »
    (réseaux de neurones TabM, RealMLP... et modèles de fondation TabICL, TabPFN, Mitra,
    TabDPT), qui demande un GPU d'au moins 20 Go ; membres agx_<modèle>. Voir
    notebooks/gpu_colab.ipynb.

AutoGluon n'est entraîné que sur D. Il écrit dans outputs/regression/library/ :
  - un membre par modèle de niveau 1 (<préfixe><modèle>) : prédictions hors bloc sur D (bagging
    d'AutoGluon), sur T et sur le test Kaggle -> utilisables par notre GES ;
  - son propre ensemble final (<préfixe>ens), qui n'est pas utilisé par l'assemblage.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.features.preprocessing import TARGET, load_locked

LIB = Path("outputs/regression/library")
EXCLUDED = ["NN_TORCH", "FASTAI", "REALMLP", "TABM", "FT_TRANSFORMER", "MITRA", "TABICL",
            "TABPFNMIX", "REALTABPFN-V2", "REALTABPFN-V2.5", "TABPFN-2.6", "TABPFN-3",
            "TABDPT", "TABDPT-TURBO", "AG_AUTOMM", "AG_TEXT_NN", "AG_IMAGE_NN"]


def save(member, family, oof, pred_T, pred_test, y_D, ymin, ymax, extra):
    oof, pred_T, pred_test = (np.clip(np.asarray(a, float), ymin, ymax) for a in (oof, pred_T, pred_test))
    LIB.mkdir(parents=True, exist_ok=True)
    np.savez(LIB / f"{member}.npz", oof=oof, pred_T=pred_T, pred_test=pred_test)
    rmse = float(np.sqrt(np.mean((np.asarray(y_D) - oof) ** 2)))
    info = dict(member=member, family=family, rmse_oof_D=rmse, **extra)
    (LIB / f"{member}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{member}] RMSE OOF(D) = {rmse:.2f}", flush=True)


if __name__ == "__main__":
    from autogluon.tabular import TabularPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=5400)
    parser.add_argument("--subsample", type=int, default=0, help="test rapide sur n lignes de D")
    parser.add_argument("--path", default="outputs/regression/autogluon/model")
    parser.add_argument("--lib", default=str(LIB))
    parser.add_argument("--preset", default="best_quality")
    parser.add_argument("--allow-nn", action="store_true",
                        help="autorise réseaux de neurones et modèles de fondation d'AutoGluon")
    parser.add_argument("--prefix", default="ag_", help="préfixe des membres écrits")
    args = parser.parse_args()
    LIB = Path(args.lib)

    X_D, y_D, X_T, y_T, X_test, _ = load_locked()
    if args.subsample:
        X_D, y_D = X_D.iloc[:args.subsample], y_D.iloc[:args.subsample]
    ymin, ymax = float(y_D.min()), float(y_D.max())
    train = X_D.copy()
    train[TARGET] = y_D.to_numpy()

    t0 = time.time()
    predictor = TabularPredictor(label=TARGET, problem_type="regression",
                                 eval_metric="root_mean_squared_error", path=args.path).fit(
        train, presets=args.preset, time_limit=args.time_limit, num_bag_folds=5,
        num_stack_levels=1, dynamic_stacking=False,
        excluded_model_types=[] if args.allow_nn else EXCLUDED)
    print(f"[autogluon] entraînement : {(time.time() - t0) / 60:.1f} min", flush=True)

    lb = predictor.leaderboard(silent=True)
    lb.to_csv(Path(args.path).parent / "leaderboard.csv", index=False)
    print(lb[["model", "score_val", "stack_level", "fit_time"]].to_string(), flush=True)

    # membres de niveau 1 (bagging -> prédictions hors bloc sur D)
    for name in lb.loc[lb["stack_level"] == 1, "model"]:
        try:
            oof = predictor.predict_oof(model=name)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] pas de prédictions hors bloc ({e})", flush=True)
            continue
        save(f"{args.prefix}{name}", "autogluon_L1", oof.to_numpy(), predictor.predict(X_T, model=name),
             predictor.predict(X_test, model=name), y_D, ymin, ymax, dict(ag_model=name))

    # ensemble final d'AutoGluon (méthode à part entière)
    best = predictor.model_best
    try:
        oof_best = predictor.predict_oof(model=best).to_numpy()
    except Exception:  # noqa: BLE001
        oof_best = np.full(len(y_D), np.nan)
    save("autogluon_ens" if args.prefix == "ag_" else f"{args.prefix}ens", "autogluon_ensemble", oof_best, predictor.predict(X_T, model=best),
         predictor.predict(X_test, model=best), y_D, ymin, ymax, dict(ag_model=best))
