"""Figures du rapport (partie régression), à partir des sorties de l'étape ensemble.

  rmse_methodes.pdf  RMSE sur T (IC 95 % bootstrap) des 10 soumissions, et score Kaggle public
  gains.pdf          chemin des gains sur T, du LightGBM de référence à la meilleure solution

Usage : python -m src.figures            (après python -m src.models.ensemble)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features.preprocessing import load_locked
from src.models.ensemble import LIB, OUT, rmse

FIG = Path("report/figures")
REF_COLOR, TOP_COLOR, KAGGLE_COLOR = "#8a8f98", "#1f4e8c", "#d1495b"

# scores publics Kaggle des soumissions déposées (relevés à la main sur le site)
KAGGLE = {"cart_1": 87.08, "knn_1": 84.44, "linear_1": 85.43, "rf_1": 81.95, "poly_1": 82.10,
          "svr_1": 82.40, "lgbm_1": 77.75, "tabicl_l2": 76.36, "tabicl_l2x": 76.23,
          "ges_jumeaux": 76.25}
LABELS = {"cart_1": "CART", "knn_1": "$k$-NN", "linear_1": "Linéaire (ridge)",
          "rf_1": "Forêt aléatoire", "poly_1": "Polynomiale", "svr_1": "SVR (Nyström)",
          "lgbm_1": "LightGBM", "tabicl_l2": "TabICL correcteur",
          "tabicl_l2x": "TabICL correcteur enrichi", "ges_jumeaux": "GES + jumeaux"}
TOP3 = {"tabicl_l2", "tabicl_l2x", "ges_jumeaux"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def fig_methodes(res):
    rows = sorted((r for r in res if r["methode"] in LABELS), key=lambda r: -r["rmse_T"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    for i, r in zip(y, rows):
        c = TOP_COLOR if r["methode"] in TOP3 else REF_COLOR
        lo, hi = r["ic95_T"]
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", alpha=0.45)
        ax.plot(r["rmse_T"], i, "o", color=c, ms=5)
        ax.plot(KAGGLE[r["methode"]], i, "x", color=KAGGLE_COLOR, ms=5, mew=1.4)
    ax.set_yticks(y, [LABELS[r["methode"]] for r in rows])
    ax.set_xlabel("RMSE (kg)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=REF_COLOR, marker="o", lw=2, label="référence : $T$, IC 95 %"),
               plt.Line2D([], [], color=TOP_COLOR, marker="o", lw=2, label="trois meilleures : $T$, IC 95 %"),
               plt.Line2D([], [], color=KAGGLE_COLOR, marker="x", ls="", mew=1.4, label="Kaggle public")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rmse_methodes.pdf")


def fig_gains(res, yT):
    by = {r["methode"]: r["rmse_T"] for r in res}
    lgbmgr_1 = rmse(yT, np.load(LIB / "lgbmgr_1.npz")["pred_T"])
    steps = [("LightGBM\nde référence", by["lgbm_1"]),
             ("groupes et\ncible normalisée", lgbmgr_1),
             ("moyenne sur\n15 graines", by["lgbmgr_bag15"]),
             ("TabICL\ncorrecteur", by["tabicl_l2"]),
             ("entrées\nAutoGluon GPU", by["tabicl_l2x"]),
             ("GES\n+ jumeaux", by["ges_jumeaux"])]
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    x = np.arange(len(steps))
    for i, (_, v) in enumerate(steps):
        if i == 0:
            ax.bar(i, v - 75.8, bottom=75.8, color=REF_COLOR, width=0.6)
            ax.text(i, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            continue
        prev = steps[i - 1][1]
        ax.bar(i, prev - v, bottom=v, color=TOP_COLOR, width=0.6)
        ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color="#999999", lw=0.7, ls=":")
        ax.text(i, prev + 0.04, f"{v - prev:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(len(steps) - 1, steps[-1][1] - 0.05, f"{steps[-1][1]:.2f}", ha="center", va="top",
            fontsize=8, fontweight="bold")
    ax.set_xticks(x, [s for s, _ in steps], fontsize=8)
    ax.set_ylim(75.8, 78.15)
    ax.set_ylabel("RMSE sur $T$ (kg)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "gains.pdf")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    yT = load_locked()[3].to_numpy(float)
    fig_methodes(res)
    fig_gains(res, yT)
    by = {r["methode"]: r["rmse_T"] for r in res}
    print(f"figures écrites dans {FIG}/ ; T :",
          {k: round(by[k], 3) for k in ("lgbm_1", "lgbmgr_bag15", "tabicl_l2", "tabicl_l2x", "ges_jumeaux")})


if __name__ == "__main__":
    main()
