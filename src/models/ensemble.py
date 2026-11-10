"""Assemblage final et évaluation sur le test T.

Méthodes comparées :
  - référence : la meilleure configuration Optuna de chaque famille classique ;
  - solutions : TabICL et LimiX-2 correcteurs des boostings (niveau 2), et l'assemblage final :
    GES (greedy ensemble selection, cours 6) sur toute la bibliothèque, puis
    correction par les « jumeaux ».
GES : moyenne avec répétitions, chaque étape ajoute le membre qui réduit le plus la RMSE de
l'ensemble courant, sur les prédictions hors bloc de D. Son erreur sur D est estimée par
validation croisée imbriquée (sélection refaite sur 4/5 des prédictions hors bloc, évaluée sur
le 1/5 restant).

Évaluation : RMSE sur T, IC 95 % par bootstrap sur les lignes de T, et bootstrap APPARIÉ des
écarts de RMSE (mêmes lignes tirées pour les deux méthodes : l'aléa commun s'annule).

Sorties : outputs/regression/ensemble/results.json et une soumission par méthode retenue
(7 références + 3 meilleures solutions) dans submissions/.

Usage : python -m src.models.ensemble
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.features.preprocessing import SEED, load_locked
from src.models.library import oof_folds

LIB = Path("outputs/regression/library")
OUT = Path("outputs/regression/ensemble")
SUB = Path("submissions")
N_BOOT = 2000
BASELINES = ["linear", "poly", "knn", "svr", "cart", "rf", "lgbm"]
REF = "lgbm_1"


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_library():
    members = {}
    for f in sorted(LIB.glob("*.npz")):
        d = np.load(f)
        info = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        # ensembles internes d'AutoGluon et membres sans prédictions hors bloc : non utilisés
        if info.get("family") == "autogluon_ensemble" or np.isnan(d["oof"]).any():
            continue
        members[f.stem] = dict(oof=d["oof"], pred_T=d["pred_T"], pred_test=d["pred_test"], **info)
    return members


def ges_weights(P, y, n_iter=50):
    """Sélection gloutonne avec remise (GES) : n_iter ajouts, chacun du membre
    qui minimise la RMSE de l'ensemble courant, même si elle remonte ; on garde l'ensemble de
    meilleure RMSE rencontré le long du chemin (poids multiples de 1/t)."""
    picks, cur, path = [], np.zeros(len(y)), []
    for _ in range(n_iter):
        scores = [rmse(y, (cur + P[:, j]) / (len(picks) + 1)) for j in range(P.shape[1])]
        j = int(np.argmin(scores))
        picks.append(j)
        cur += P[:, j]
        path.append(scores[j])
    t = int(np.argmin(path)) + 1
    w = np.bincount(picks[:t], minlength=P.shape[1]).astype(float)
    return w / w.sum()


def nested_cv(fit_weights, P, y, n_outer=5):
    """RMSE hors bloc de la PROCÉDURE d'assemblage (choix des poids refait à chaque bloc)."""
    pred = np.zeros(len(y))
    for tr, va in KFold(n_outer, shuffle=True, random_state=SEED + 1).split(P):
        pred[va] = P[va] @ fit_weights(P[tr], y[tr])
    return rmse(y, pred)


def twin_values(X_ref, y_ref, X_new, folds=None):
    """Total moyen des « jumeaux » (même valeur sur les 16 variables d'origine) de chaque ligne
    de X_new parmi les lignes de référence étiquetées ; NaN sans jumeau. Avec `folds` (X_new =
    X_ref = D), chaque ligne n'est comparée qu'aux lignes des AUTRES blocs (hors bloc)."""
    key_ref, key_new = X_ref["twin"].to_numpy(), X_new["twin"].to_numpy()
    y_ref = np.asarray(y_ref, float)
    if folds is None:
        return pd.Series(key_new).map(pd.Series(y_ref).groupby(key_ref).mean()).to_numpy()
    out = np.full(len(key_new), np.nan)
    for tr, va in folds:
        out[va] = pd.Series(key_new[va]).map(pd.Series(y_ref[tr]).groupby(key_ref[tr]).mean()).to_numpy()
    return out


def apply_twins(pred, twin, alpha):
    """p' = p + alpha (jumeau - p) là où un jumeau existe."""
    out = np.array(pred, float)
    has = ~np.isnan(twin)
    out[has] += alpha * (twin[has] - out[has])
    return out


def bootstrap(y, preds, ref, n_boot=N_BOOT):
    """IC 95 % de la RMSE de chaque méthode et de l'écart apparié à la méthode `ref`."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    sq = {k: (y - p) ** 2 for k, p in preds.items()}
    out = {}
    for k in preds:
        r = np.sqrt(sq[k][idx].mean(axis=1))
        d = r - np.sqrt(sq[ref][idx].mean(axis=1))
        out[k] = dict(ci=np.percentile(r, [2.5, 97.5]).tolist(),
                      delta=float(rmse(y, preds[k]) - rmse(y, preds[ref])),
                      delta_ci=np.percentile(d, [2.5, 97.5]).tolist())
    return out


def main():
    X_D, y_D, X_T, y_T, X_test, test_ids = load_locked()
    yD, yT = y_D.to_numpy(float), y_T.to_numpy(float)
    lib = load_library()
    members = list(lib)
    print(f"{len(members)} membres dans la bibliothèque")

    methods = {}   # nom -> dict(cv_D, pred_T, pred_test, weights)
    for m in [f"{f}_1" for f in BASELINES] + ["lgbmgr_bag15", "catgr_1", "tabicl", "tabicl_l2",
                                               "tabicl_l2x", "limix_l2"]:
        if m in lib:
            methods[m] = dict(cv_D=lib[m]["rmse_oof_D"], pred_T=lib[m]["pred_T"],
                              pred_test=lib[m]["pred_test"])

    # GES sur toute la bibliothèque (poids choisis sur les prédictions hors bloc de D)
    P = np.column_stack([lib[m]["oof"] for m in members])
    w = ges_weights(P, yD)
    pred_T = np.column_stack([lib[m]["pred_T"] for m in members]) @ w
    pred_test = np.column_stack([lib[m]["pred_test"] for m in members]) @ w
    weights = {m: round(float(x), 4) for m, x in zip(members, w) if x > 0}
    methods["ges"] = dict(cv_D=nested_cv(ges_weights, P, yD), pred_T=pred_T, pred_test=pred_test,
                          weights=weights)

    # correction par les jumeaux (alpha choisi sur D, jumeaux cherchés hors bloc)
    tw_D = twin_values(X_D, yD, X_D, oof_folds(len(yD)))
    tw_T = twin_values(X_D, yD, X_T)
    X_all, y_all = pd.concat([X_D, X_T], ignore_index=True), np.concatenate([yD, yT])
    tw_test = twin_values(X_all, y_all, X_test)
    oof_ges = P @ w
    alphas = np.linspace(0, 1, 11)
    alpha = float(alphas[np.argmin([rmse(yD, apply_twins(oof_ges, tw_D, a)) for a in alphas])])
    methods["ges_jumeaux"] = dict(
        cv_D=rmse(yD, apply_twins(oof_ges, tw_D, alpha)),
        pred_T=apply_twins(pred_T, tw_T, alpha), pred_test=apply_twins(pred_test, tw_test, alpha),
        weights={**weights, "alpha_jumeaux": alpha})
    print(f"GES : {weights}\njumeaux : alpha = {alpha:.1f}")

    boot = bootstrap(yT, {k: v["pred_T"] for k, v in methods.items()}, REF)
    rows = [dict(methode=k, rmse_cv_D=v["cv_D"], rmse_T=rmse(yT, v["pred_T"]), ic95_T=boot[k]["ci"],
                 delta_vs_ref=boot[k]["delta"], ic95_delta=boot[k]["delta_ci"],
                 poids=v.get("weights")) for k, v in methods.items()]
    res = pd.DataFrame(rows).sort_values("rmse_T")
    pd.set_option("display.width", 250)
    print(res.drop(columns="poids").round(2).to_string(index=False))

    # écarts appariés entre solutions successives
    pairs = [("ges_jumeaux", "tabicl_l2x"), ("tabicl_l2x", "tabicl_l2"), ("ges_jumeaux", "tabicl_l2"),
             ("tabicl_l2", "lgbmgr_bag15"), ("limix_l2", "lgbmgr_bag15")]
    paired = {}
    for a, b in pairs:
        if a in methods and b in methods:
            d = bootstrap(yT, {a: methods[a]["pred_T"], b: methods[b]["pred_T"]}, b)[a]
            paired[f"{a} - {b}"] = dict(delta=d["delta"], ic95=d["delta_ci"])
            print(f"{a} - {b} : {d['delta']:+.2f} [{d['delta_ci'][0]:+.2f} ; {d['delta_ci'][1]:+.2f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_json(OUT / "results.json", orient="records", indent=2, force_ascii=False)
    (OUT / "paired.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")

    # soumissions : les 7 références et les 3 meilleures solutions (classées sur D)
    solutions = sorted([m for m in ["ges_jumeaux", "tabicl_l2x", "tabicl_l2", "limix_l2"]
                        if m in methods], key=lambda m: methods[m]["cv_D"])[:3][::-1]
    SUB.mkdir(exist_ok=True)
    for old in SUB.glob("*.csv"):
        old.unlink()
    # numérotées de la plus faible à la meilleure : 01 = CART, ..., dernière = meilleure solution
    order = [f"{f}_1" for f in sorted(BASELINES, key=lambda f: -methods[f"{f}_1"]["cv_D"])] + solutions
    for i, k in enumerate(order, 1):
        pd.DataFrame({"id": test_ids, "TotalKg": methods[k]["pred_test"]}).to_csv(
            SUB / f"{i:02d}_{k}.csv", index=False)
    print(f"{len(BASELINES) + len(solutions)} soumissions écrites dans {SUB}/")


if __name__ == "__main__":
    main()
