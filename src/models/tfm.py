"""Modèles de fondation tabulaires (TabICL, LimiX-2) comme membres de la bibliothèque.

Le modèle, pré-entraîné sur des jeux synthétiques, reçoit un CONTEXTE de
lignes étiquetées et prédit les autres lignes en une passe avant (in-context learning).

Entrées : variables d'origine, effectifs de groupe, clés de groupe ENCODÉES PAR LA CIBLE sur le
pool (validation croisée interne), et, au « niveau 2 » (--base), les prédictions hors bloc de
membres de la bibliothèque : le modèle apprend en contexte où et comment les corriger.

7 passes par membre : 5 blocs hors bloc sur D (pool = D privé du bloc, les mêmes blocs que la
bibliothèque), T (pool = D) et le test Kaggle (pool = D ∪ T). Chaque passe est sauvegardée :
une exécution interrompue reprend où elle s'était arrêtée.

Taille du contexte : sur GPU, --ctx auto (défaut) lance une SONDE qui mesure le temps et le pic
de mémoire de vrais appels et retient le plus grand contexte qui tient en mémoire et dans le
budget de temps. Sur CPU, le contexte par défaut est de 16 000 lignes.

Usage :
  python -m src.models.tfm --model tabicl --name tabicl
  python -m src.models.tfm --model tabicl --name tabicl_l2 --base lgbmgr_bag15 catgr_1 ag_LightGBMXT_BAG_L1
  python -m src.models.tfm --model limix  --name limix_l2  --base lgbmgr_bag15 catgr_1 ag_LightGBMXT_BAG_L1
LimiX-2 demande le dépôt https://github.com/limix-ldm-ai/LimiX (--limix-dir).
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from src.features.preprocessing import GROUP_KEYS, GROUP_NUM, RAW_VARS, SEED, load_locked
from src.models.library import LIB, oof_folds, save_member

warnings.filterwarnings("ignore")
PARTS = Path("outputs/regression/tfm_parts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else float("inf")
MEM_SAFETY = 0.85


# ------------------------------------------------------------------------------ données
def make_frames(A_raw, y_pool, B_raw, base_pool, base_pred, numeric=False):
    """Variables d'origine + effectifs + encodages par la cible (+ prédictions de base)."""
    te = TargetEncoder(target_type="continuous", random_state=SEED)
    E_pool = te.fit_transform(A_raw[GROUP_KEYS], np.asarray(y_pool))   # validation croisée interne
    E_pred = te.transform(B_raw[GROUP_KEYS])
    A = A_raw[RAW_VARS + GROUP_NUM].reset_index(drop=True).copy()
    B = B_raw[RAW_VARS + GROUP_NUM].reset_index(drop=True).copy()
    for j, k in enumerate(GROUP_KEYS):
        A[f"te_{k}"], B[f"te_{k}"] = E_pool[:, j], E_pred[:, j]
    for j in range(base_pool.shape[1]):
        A[f"base_{j}"], B[f"base_{j}"] = base_pool[:, j], base_pred[:, j]
    if numeric:     # matrice numérique (LimiX) : catégorielles en codes entiers, sans la cible
        cat = [c for c in RAW_VARS if A[c].dtype == object]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(pd.concat([A[cat], B[cat]]))
        A[cat], B[cat] = enc.transform(A[cat]), enc.transform(B[cat])
        A, B = A.astype(np.float32).to_numpy(), B.astype(np.float32).to_numpy()
    return A, B


def passes(bases):
    """Les 7 passes : (étiquette, X_pool, y_pool, base_pool, X_pred, base_pred)."""
    X_D, y_D, X_T, y_T, X_test, _ = load_locked()
    B = [np.load(LIB / f"{m}.npz") for m in bases]
    col = lambda key: np.column_stack([b[key] for b in B]) if B else np.zeros((0, 0))
    oof, pT, ptest = col("oof"), col("pred_T"), col("pred_test")
    sel = lambda a, idx: a[idx] if B else np.zeros((len(idx), 0))
    out = []
    for k, (tr, va) in enumerate(oof_folds(len(y_D))):
        out.append((f"fold{k}", X_D.iloc[tr], y_D.iloc[tr], sel(oof, tr), X_D.iloc[va], sel(oof, va)))
    out.append(("T", X_D, y_D, oof if B else np.zeros((len(y_D), 0)), X_T,
                pT if B else np.zeros((len(X_T), 0))))
    X_all, y_all = pd.concat([X_D, X_T], ignore_index=True), pd.concat([y_D, y_T], ignore_index=True)
    # pour T, la prédiction « hors échantillon » d'un membre est pred_T (modèle appris sur D)
    out.append(("test", X_all, y_all, np.vstack([oof, pT]) if B else np.zeros((len(y_all), 0)),
                X_test, ptest if B else np.zeros((len(X_test), 0))))
    return out, X_D, y_D


def measure(fn):
    """(succès, secondes, pic de mémoire GPU en Go) ; un manque de mémoire -> succès = False."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0, ok = time.time(), True
    try:
        fn()
    except Exception as e:
        if "out of memory" not in str(e).lower():
            raise
        ok = False
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return ok, dt, peak


def subsample(n, size, seed):
    return np.arange(n) if size >= n else np.random.default_rng(seed).choice(n, size, replace=False)


# ------------------------------------------------------------------------------ TabICL
class TabICLBackend:
    numeric = False

    def __init__(self, args):
        from tabicl import TabICLRegressor
        self.cls, self.args = TabICLRegressor, args

    def model(self, seed):
        return self.cls(device=DEVICE, n_estimators=self.args.n_estimators, kv_cache=True,
                        random_state=seed)

    def predict(self, A, y, B, n_ctx):
        draws = 1 if n_ctx >= len(A) else self.args.draws
        out = np.zeros(len(B))
        for j in range(draws):          # tirages de contexte moyennés si contexte < pool
            idx = subsample(len(A), n_ctx, SEED + 1000 * j)
            m = self.model(SEED + 1000 * j).fit(A.iloc[idx], y[idx])
            out += np.concatenate([m.predict(B.iloc[i:i + self.args.chunk])
                                   for i in range(0, len(B), self.args.chunk)]) / draws
            del m; gc.collect()
        return out

    def probe(self, frames, sizes):
        """Plus grand contexte (parmi les candidats) qui tient en mémoire et dans le budget."""
        A, B, y = frames
        self.model(0).fit(A.iloc[:1000], y[:1000]).predict(B.iloc[:100])     # préchauffage
        best = None
        for N in [105000, 84000, 67200, 50000, 40000, 30000, 16000]:
            idx = subsample(len(A), N, 1)
            hold = {}
            ok1, t_fit, p1 = measure(lambda: hold.setdefault("m", self.model(0).fit(A.iloc[idx], y[idx])))
            ok2, t_chunk, p2 = (measure(lambda: hold["m"].predict(B.iloc[:self.args.chunk]))
                                if ok1 else (False, 0, 0))
            hold.clear()
            if not (ok1 and ok2) or max(p1, p2) > MEM_SAFETY * VRAM:
                print(f"  contexte {N:6d} : mémoire insuffisante", flush=True)
                continue
            est = 0.0
            for pool, n_pred in sizes:
                n_eff = min(N, pool); draws = 1 if n_eff >= pool else self.args.draws
                est += draws * (t_fit + math.ceil(n_pred / self.args.chunk) * t_chunk) * n_eff / N
            print(f"  contexte {N:6d} : {t_fit:.1f}s + {t_chunk:.1f}s/morceau, pic "
                  f"{max(p1, p2):.1f} Go -> {est / 3600:.2f} h", flush=True)
            best = best or N
            if est <= self.args.budget_h * 3600:
                return N
        if best is None:
            raise RuntimeError("aucun contexte ne tient en mémoire : réduire --chunk")
        return 16000


# ------------------------------------------------------------------------------ LimiX-2
class LimiXBackend:
    """LimiX ne met pas le contexte en cache : chaque morceau retraite tout le contexte, pour
    chaque pipeline de pré-traitement de la config (8 par défaut ; on en garde --limix-pipelines)."""
    numeric = True

    def __init__(self, args):
        sys.path.insert(0, args.limix_dir)
        os.environ.update(RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
        from huggingface_hub import hf_hub_download
        from inference.predictor import LimiXPredictor
        ckpt = None
        for repo in ("stable-ai/LimiX-2", "stableai-org/LimiX-2"):
            try:
                ckpt = hf_hub_download(repo_id=repo, filename="LimiX-2.ckpt", local_dir=".cache/limix")
                break
            except Exception as e:
                print(repo, "->", type(e).__name__)
        cfg = json.loads(Path(args.limix_dir, "config", "reg_default_noretrieval_v2.json").read_text())
        cfg["pipelines"] = cfg["pipelines"][:args.limix_pipelines]
        Path(".cache").mkdir(exist_ok=True)
        Path(".cache/limix_cfg.json").write_text(json.dumps(cfg))
        self.model = LimiXPredictor(device=torch.device(DEVICE), model_path=ckpt,
                                    inference_config=".cache/limix_cfg.json")
        self.args, self.chunk = args, 4000

    def call(self, A, y, B):
        return np.asarray(self.model.predict(A, y, B, task_type="Regression")).ravel()

    def predict(self, A, y, B, n_ctx):
        idx = subsample(len(A), n_ctx, SEED)
        return np.concatenate([self.call(A[idx], y[idx], B[i:i + self.chunk])
                               for i in range(0, len(B), self.chunk)])

    def probe(self, frames, sizes):
        """Temps t = a + b(N+m) + cN(N+m) et mémoire M = m0 + m1(N+m) + m2(N+m)^2 ajustés sur
        4 mesures, choix du plus grand contexte N (puis du morceau m le plus rapide) dans le
        budget, puis vérification sur un vrai morceau (recalage si l'extrapolation se trompe)."""
        A, B, y = frames
        self.call(A[:1000], y[:1000], B[:200])                              # préchauffage
        Xt, Xm, ts, ms = [], [], [], []
        for N, m in [(4000, 1000), (8000, 1000), (8000, 4000), (16000, 2000)]:
            ok, dt, peak = measure(lambda: self.call(A[:N], y[:N], B[:m]))
            if not ok:
                raise RuntimeError(f"mémoire insuffisante dès N={N}")
            print(f"  N={N:6d} m={m:5d} : {dt:6.1f}s, pic {peak:5.1f} Go", flush=True)
            Xt.append([1, N + m, N * (N + m)]); ts.append(dt)
            Xm.append([1, N + m, (N + m) ** 2]); ms.append(peak)
        ct = np.clip(np.linalg.lstsq(np.array(Xt, float), ts, rcond=None)[0], 0, None)
        cm = np.clip(np.linalg.lstsq(np.array(Xm, float), ms, rcond=None)[0], 0, None)
        mem = lambda N, m: cm[0] + cm[1] * (N + m) + cm[2] * (N + m) ** 2
        grid = list(range(5000, 105001, 5000))
        scale, best = 1.0, None
        for _ in range(3):
            t = lambda N, m: scale * (ct[0] + ct[1] * (N + m) + ct[2] * N * (N + m))
            total = lambda N, m: sum(math.ceil(n / m) * t(min(N, p), m) for p, n in sizes)
            cand = [(N, m, total(N, m)) for N in grid for m in (1000, 2000, 4000, 8000, 16000)
                    if mem(N, m) < MEM_SAFETY * VRAM]
            fit = [c for c in cand if c[2] <= self.args.budget_h * 3600]
            N, m, _ = max(fit, key=lambda c: (c[0], -c[2])) if fit else min(cand, key=lambda c: c[2])
            ok, dt, peak = measure(lambda: self.call(A[subsample(len(A), N, 1)],
                                                     y[subsample(len(A), N, 1)], B[:m]))
            print(f"  vérification N={N} m={m} : {dt:.0f}s mesurées / {t(N, m):.0f}s prévues "
                  f"-> {total(N, m) / 3600:.2f} h", flush=True)
            if not ok:
                grid = [n for n in grid if n < N]; continue
            best = (N, m)
            if dt <= 1.25 * t(N, m):
                break
            scale *= dt / t(N, m)
        if best is None:
            raise RuntimeError("aucun point vérifié ne tient en mémoire")
        self.chunk = best[1]
        return best[0]


# ------------------------------------------------------------------------------ principal
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["tabicl", "limix"], default="tabicl")
    p.add_argument("--name", required=True, help="nom du membre de la bibliothèque")
    p.add_argument("--base", nargs="*", default=[], help="membres ajoutés en entrée (niveau 2)")
    p.add_argument("--ctx", default="auto" if DEVICE == "cuda" else "16000",
                   help="taille du contexte, ou 'auto' (sonde)")
    p.add_argument("--budget-h", type=float, default=1.5, help="budget de temps du membre (sonde)")
    p.add_argument("--n-estimators", type=int, default=8 if DEVICE == "cuda" else 1)
    p.add_argument("--draws", type=int, default=2, help="tirages de contexte si contexte < pool")
    p.add_argument("--chunk", type=int, default=5000, help="lignes prédites par appel (TabICL)")
    p.add_argument("--limix-dir", default=os.environ.get("LIMIX_DIR", "LimiX"))
    p.add_argument("--limix-pipelines", type=int, default=2)
    args = p.parse_args()

    print(f"[{args.name}] {args.model} sur {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)}, {VRAM:.0f} Go)" if DEVICE == "cuda" else ""))
    backend = (TabICLBackend if args.model == "tabicl" else LimiXBackend)(args)
    todo, X_D, y_D = passes(args.base)
    sizes = [(len(t[1]), len(t[4])) for t in todo]

    if args.ctx == "auto":
        tag, X_pool, y_pool, b_pool, X_pred, b_pred = todo[-1]       # le plus grand pool
        A, B = make_frames(X_pool, y_pool, X_pred.iloc[:20000], b_pool, b_pred[:20000],
                           backend.numeric)
        print("Sonde :", flush=True)
        n_ctx = backend.probe((A, B, np.asarray(y_pool)), sizes)
    else:
        n_ctx = int(args.ctx)
    print(f"=> contexte {n_ctx}", flush=True)

    parts = PARTS / args.name
    parts.mkdir(parents=True, exist_ok=True)
    res, t0 = {}, time.time()
    for tag, X_pool, y_pool, b_pool, X_pred, b_pred in todo:
        f = parts / f"{tag}.npy"
        if not f.exists():
            A, B = make_frames(X_pool, y_pool, X_pred, b_pool, b_pred, backend.numeric)
            np.save(f, backend.predict(A, np.asarray(y_pool), B, n_ctx))
        res[tag] = np.load(f)
        print(f"[{args.name}] {tag} : {(time.time() - t0) / 60:.1f} min", flush=True)

    oof = np.zeros(len(y_D))
    for k, (_, va) in enumerate(oof_folds(len(y_D))):
        oof[va] = res[f"fold{k}"]
    lo, hi = float(y_D.min()), float(y_D.max())
    r = save_member(args.name, args.model, dict(model=args.model, base=args.base, n_ctx=n_ctx,
                                                n_estimators=args.n_estimators, device=DEVICE),
                    np.clip(oof, lo, hi), np.clip(res["T"], lo, hi), np.clip(res["test"], lo, hi),
                    y_D)
    print(f"[{args.name}] RMSE OOF(D) = {r:.2f}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
