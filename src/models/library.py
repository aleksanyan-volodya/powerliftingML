"""Bibliothèque de configurations (la philosophie : « ne pas jeter les modèles perdants »).

Pour chaque famille de modèles, les k meilleures configurations distinctes de l'étude Optuna
(recherche par validation croisée sur D) deviennent des membres de la bibliothèque. Chaque
membre est stocké dans outputs/regression/library/<membre>.npz avec :
  oof        prédictions hors bloc sur D (5 blocs, les mêmes pour tous les membres)
  pred_T     prédictions sur le test verrouillé T (modèle appris sur tout D)
  pred_test  prédictions sur le test Kaggle (modèle ré-appris sur D ∪ T)
Les membres externes (TabICL, AutoGluon) sont écrits au même format par leurs scripts.

Usage : python -m src.models.library [familles...] [--top-k 3]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from src.features.preprocessing import SEED, load_locked
from src.models.zoo import MODELS

OUT = Path("outputs/regression")
LIB = OUT / "library"
N_FOLDS = 5


def oof_folds(n):
    """Blocs de la validation croisée hors bloc, communs à tous les membres."""
    return list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)))


def top_configs(name, k, pick=None):
    """Les k meilleurs essais terminés et distincts de l'étude Optuna de la famille `name`
    (ou, si `pick` est donné, les essais de ces numéros, dans cet ordre)."""
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{OUT / name / 'study.db'}")
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if pick:
        by_number = {t.number: t for t in done}
        return [by_number[n] for n in pick]
    done.sort(key=lambda t: t.value)
    out, seen = [], set()
    for t in done:
        key = json.dumps(t.params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(t)
        if len(out) == k:
            break
    return out


def save_member(member, family, params, oof, pred_T, pred_test, y_D, extra=None):
    LIB.mkdir(parents=True, exist_ok=True)
    np.savez(LIB / f"{member}.npz", oof=oof, pred_T=pred_T, pred_test=pred_test)
    info = dict(member=member, family=family, params=params,
                rmse_oof_D=float(root_mean_squared_error(y_D, oof)))
    info.update(extra or {})
    (LIB / f"{member}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return info["rmse_oof_D"]


def build_family(name, k, X_D, y_D, X_T, y_T, X_test, pick=None):
    spec = MODELS[name]
    X_all = pd.concat([X_D, X_T], ignore_index=True)
    y_all = pd.concat([y_D, y_T], ignore_index=True)
    ymin, ymax = float(y_D.min()), float(y_D.max())
    folds = oof_folds(len(y_D))
    for rank, trial in enumerate(top_configs(name, k, pick), start=1):
        member = f"{name}_{rank}"
        params = spec.space(optuna.trial.FixedTrial(trial.params))
        t0 = time.time()
        oof = np.zeros(len(y_D))
        for tr, va in folds:
            oof[va] = spec.build(params).fit(X_D.iloc[tr], y_D.iloc[tr]).predict(X_D.iloc[va])
        pred_T = spec.build(params).fit(X_D, y_D).predict(X_T)            # T jamais vu
        pred_test = spec.build(params).fit(X_all, y_all).predict(X_test)  # soumission
        r = save_member(member, name, trial.params, np.clip(oof, ymin, ymax),
                        np.clip(pred_T, ymin, ymax), np.clip(pred_test, ymin, ymax), y_D,
                        dict(optuna_trial=trial.number, rmse_cv3_D=trial.value))
        print(f"[{member}] RMSE OOF(D) = {r:6.2f}  (essai Optuna {trial.number}, "
              f"CV3 = {trial.value:.2f})  {time.time() - t0:.0f}s", flush=True)


def build_bagged(name, n_seeds, X_D, y_D, X_T, y_T, X_test):
    """Moyenne de la meilleure configuration sur `n_seeds` graines (sous-échantillonnage des
    lignes et colonnes, découpage d'arrêt anticipé)."""
    spec = MODELS[name]
    trial = top_configs(name, 1)[0]
    X_all = pd.concat([X_D, X_T], ignore_index=True)
    y_all = pd.concat([y_D, y_T], ignore_index=True)
    ymin, ymax = float(y_D.min()), float(y_D.max())
    folds = oof_folds(len(y_D))
    oof, pred_T, pred_test = np.zeros(len(y_D)), np.zeros(len(X_T)), np.zeros(len(X_test))
    t0 = time.time()
    for s in range(n_seeds):
        params = dict(spec.space(optuna.trial.FixedTrial(trial.params)), random_state=SEED + 100 * s)
        for tr, va in folds:
            oof[va] += spec.build(params).fit(X_D.iloc[tr], y_D.iloc[tr]).predict(X_D.iloc[va]) / n_seeds
        pred_T += spec.build(params).fit(X_D, y_D).predict(X_T) / n_seeds
        pred_test += spec.build(params).fit(X_all, y_all).predict(X_test) / n_seeds
        print(f"[{name}_bag{n_seeds}] graine {s + 1}/{n_seeds} : RMSE OOF partielle "
              f"{root_mean_squared_error(y_D, oof * n_seeds / (s + 1)):.2f}  {time.time() - t0:.0f}s",
              flush=True)
    r = save_member(f"{name}_bag{n_seeds}", name, trial.params, np.clip(oof, ymin, ymax),
                    np.clip(pred_T, ymin, ymax), np.clip(pred_test, ymin, ymax), y_D,
                    dict(optuna_trial=trial.number, n_seeds=n_seeds))
    print(f"[{name}_bag{n_seeds}] RMSE OOF(D) = {r:6.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("families", nargs="*", default=list(MODELS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bag-seeds", type=int, default=0,
                        help="en plus, moyenne de la meilleure configuration sur n graines")
    parser.add_argument("--pick", type=int, nargs="*", default=None,
                        help="numéros d'essais Optuna à utiliser au lieu des k meilleurs")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_D, y_D, X_T, y_T, X_test, _ = load_locked()
    for fam in args.families:
        if args.top_k or args.pick:
            build_family(fam, args.top_k, X_D, y_D, X_T, y_T, X_test, args.pick)
        if args.bag_seeds:
            build_bagged(fam, args.bag_seeds, X_D, y_D, X_T, y_T, X_test)
