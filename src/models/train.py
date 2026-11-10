"""Recherche d'hyperparamètres (Optuna, validation croisée sur D ; T reste verrouillé).

Usage : python -m src.models.train linear knn ...   [--trials N] [--timeout SEC]

Pour chaque modèle, écrit dans outputs/regression/<modèle>/ :
  study.db          étude Optuna (reprise possible en relançant la commande)
  best_params.json  meilleurs hyperparamètres + RMSE de validation croisée sur D

Cette étape ne choisit QUE les hyperparamètres. Les prédictions hors bloc (bibliothèque de
configurations), l'assemblage et l'évaluation sur T sont faits par src.models.stacking.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from src.features.preprocessing import SEED, load_locked
from src.models.zoo import MODELS

OUT = Path("outputs/regression")
N_FOLDS = 3   # recherche d'hyperparamètres : 3 blocs (rapide) ; le stacking final en utilise 5


def objective_cv(spec, X, y, folds):
    def objective(trial):
        params = spec.space(trial)
        scores = []
        for k, (tr, va) in enumerate(folds):
            model = spec.build(params).fit(X.iloc[tr], y.iloc[tr])
            scores.append(root_mean_squared_error(y.iloc[va], model.predict(X.iloc[va])))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():          # arrêt anticipé des essais clairement mauvais
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def run(name, n_trials=None, timeout=None, warm_from=None, warm_k=3, warm_extra=None):
    spec = MODELS[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    X, y, *_ = load_locked()          # D uniquement
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X))

    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{out / 'study.db'}", load_if_exists=True,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    if warm_from and not study.trials:
        # démarrage à chaud : meilleurs essais d'une autre étude (elle aussi évaluée sur D seul)
        src = optuna.load_study(study_name=warm_from,
                                storage=f"sqlite:///{OUT / warm_from / 'study.db'}")
        done = sorted([t for t in src.trials if t.state == optuna.trial.TrialState.COMPLETE],
                      key=lambda t: t.value)[:warm_k]
        for t in done:
            study.enqueue_trial({**t.params, **(warm_extra or {})})
        print(f"[{name}] {len(done)} essais de {warm_from} mis en file", flush=True)
    n_done = len([t for t in study.trials if t.state.is_finished()])
    n_todo = max(0, (n_trials or spec.n_trials) - n_done)
    t0 = time.time()
    if n_todo:
        study.optimize(objective_cv(spec, X, y, folds), n_trials=n_todo, timeout=timeout,
                       callbacks=[lambda st, tr: print(
                           f"[{name}] essai {tr.number:3d} {tr.state.name:8s} "
                           f"RMSE CV = {tr.value if tr.value is not None else float('nan'):7.2f}"
                           f"  meilleur = {st.best_value:7.2f}", flush=True)])
    best = study.best_trial
    print(f"[{name}] optimisation : {(time.time() - t0) / 60:.1f} min, "
          f"meilleur RMSE CV = {best.value:.2f}, params = {best.params}", flush=True)

    (out / "best_params.json").write_text(json.dumps(
        dict(model=name, label=spec.label, params=best.params, rmse_cv=best.value,
             n_trials=len(study.trials),
             n_pruned=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])),
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=list(MODELS))
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="secondes par modèle")
    parser.add_argument("--warm-from", default=None, help="étude dont on reprend les meilleurs essais")
    parser.add_argument("--warm-k", type=int, default=3)
    parser.add_argument("--warm-extra", default=None, help="paramètres ajoutés, ex. bw_power=0.8")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    extra = None
    if args.warm_extra:
        extra = {k: float(v) for k, v in (kv.split("=") for kv in args.warm_extra.split(","))}
    for name in args.models:
        run(name, args.trials, args.timeout, args.warm_from, args.warm_k, extra)
