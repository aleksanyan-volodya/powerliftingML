# Powerlifting ML - Data challenge Régression

Il s'agit de prédire le total soulevé (`TotalKg`) en powerlifting à partir
des caractéristiques des athlètes et des compétitions.

## Structure du dépôt

```
.
├── data/
│   └── raw/                  # train.csv et test.csv Kaggle (non versionnés)
├── notebooks/
│   └── gpu_colab.ipynb       # exécution de l'étape GPU sur Colab
├── src/
│   ├── features/
│   │   └── preprocessing.py  # variables, pré-traitements, découpage D/T
│   ├── models/               # espaces de recherche, entraînement, bibliothèque, assemblage
│   ├── figures.py            # figures du rapport
│   ├── gpu_bundle.py         # bundle pour Colab
│   └── pipeline.py           # enchaînement des étapes
├── outputs/                  # sorties intermédiaires (non versionnées)
├── submissions/              # fichiers de soumission Kaggle (non versionnés)
├── pyproject.toml            # métadonnées et dépendances
└── README.md
```

## Installation

Prérequis : Python 3.13+ et Git.

```bash
git clone https://github.com/aleksanyan-volodya/powerliftingML
cd powerliftingML
pip install -e .
```

Ou avec `uv` (plus rapide) :
```bash
uv sync
```

## Données

Télécharger les données Kaggle dans `data/raw/` :
```bash
kaggle competitions download -c performances-en-powerlifting -p data/raw/
```

## Soumissions

`python -m src.models.ensemble` écrit les fichiers Kaggle dans `submissions/` (`01_cart_1.csv` …
`10_ges_jumeaux.csv`, format `id,TotalKg`).

## Pipeline de régression (TotalKg)

Données : `data/raw/train.csv` et `data/raw/test.csv`. Le train est séparé une fois (graine
fixée) en **D** (développement, 84 000 lignes) et **T** (test verrouillé, 21 000 lignes). Tous
les choix se font sur D par validation croisée ; T sert à estimer l'erreur de chaque méthode.
Les soumissions sont ré-entraînées sur D ∪ T.

### Lancer le pipeline

```bash
pip install -e .
python -m src.pipeline                      # toutes les étapes
python -m src.pipeline --steps ensemble     # ou une partie : search library autogluon gpu ensemble
```

| Étape | Contenu | Durée indicative (CPU 8 cœurs) |
|---|---|---|
| `search` | Optuna (validation croisée à 3 blocs sur D) : 7 familles de référence, puis LightGBM et CatBoost structurés | ~6 h |
| `library` | prédictions hors bloc sur D (5 blocs), sur T et sur le test, pour les meilleures configurations | ~3 h |
| `autogluon` | AutoGluon (`best_quality`, 1,5 h), environnement séparé (voir ci-dessous) | 1,5 h |
| `gpu` | AutoGluon `extreme` (réseaux de neurones et modèles de fondation, 4 h), TabICL et LimiX-2 : en local s'il y a un GPU, sinon via Colab (voir ci-dessous) | GPU : ~7 h |
| `ensemble` | assemblage GES + jumeaux, évaluation sur T, soumissions | quelques minutes |

LightGBM et CatBoost occupent tous les cœurs : ne pas lancer deux étapes en parallèle.

**AutoGluon** ne s'installe pas encore sous Python ≥ 3.13 :
```bash
uv venv .venv-autogluon --python 3.12 --native-tls
VIRTUAL_ENV=.venv-autogluon uv pip install --native-tls -e . "autogluon.tabular[all,tabarena]"
```
Les étapes `autogluon` et `gpu` utilisent alors `.venv-autogluon` (option `--autogluon-python`
sinon). Les réseaux de neurones ne sont utilisés qu'à travers AutoGluon, sur GPU (preset
`extreme`, au moins 20 Go de mémoire GPU).

**Sans GPU local.** L'étape `gpu` crée `gpu_bundle.zip` (code, données, prédictions des
boostings), puis s'arrête.
1. Ouvrir [notebooks/gpu_colab.ipynb](notebooks/gpu_colab.ipynb) dans
   Colab, avec un GPU A100 ou L4 et la mémoire RAM élevée.
2. Choisir les sections à exécuter (cellule 1), uploader `gpu_bundle.zip` et lancer
   *Tout exécuter*.
3. Décompresser le `gpu_results.zip` obtenu à la racine du dépôt.
4. Lancer `python -m src.pipeline --steps ensemble`.

Le notebook exécute les mêmes commandes qu'en local. Pour TabICL et LimiX-2, une sonde
choisit la taille du contexte selon la mémoire du GPU et le budget de temps.

### Résultats

RMSE (kg) sur le test verrouillé T et score Kaggle public. Les soumissions sont dans
`submissions/`, numérotées de la plus faible à la meilleure.

| N° | Méthode | T | Kaggle |
|---|---|---|---|
| 01 | CART | 88.17 | 87.08 |
| 02 | k plus proches voisins | 85.00 | 84.44 |
| 03 | Régression linéaire (ridge) | 85.00 | 85.43 |
| 04 | Forêt aléatoire | 82.09 | 81.95 |
| 05 | Régression polynomiale | 81.74 | 82.10 |
| 06 | SVR (Nyström) | 82.05 | 82.40 |
| 07 | LightGBM | 77.87 | 77.75 |
| 08 | **TabICL, correcteur des boostings** (`tabicl_l2`) | 76.35 | 76.36 |
| 09 | **TabICL, correcteur enrichi des modèles AutoGluon GPU** (`tabicl_l2x`) | 76.28 | **76.23** |
| 10 | **Assemblage GES + jumeaux** (`ges_jumeaux`) | **76.23** | 76.25 |

### Code

| Fichier | Rôle |
|---|---|
| `src/features/preprocessing.py` | variables, variables de groupe, pré-traitements, découpage D/T |
| `src/models/zoo.py` | espaces de recherche Optuna et pipelines des familles |
| `src/models/train.py` | recherche d'hyperparamètres |
| `src/models/library.py` | membres de la bibliothèque (prédictions hors bloc, T, test) |
| `src/models/autogluon_run.py` | AutoGluon (CPU, ou GPU avec réseaux de neurones) |
| `src/models/tfm.py` | TabICL et LimiX-2 (sonde de contexte, reprise après interruption) |
| `src/models/ensemble.py` | assemblage, évaluation sur T, soumissions |
| `src/figures.py` | figures du rapport (`python -m src.figures`, après l'assemblage) |
| `src/pipeline.py` | enchaînement des étapes |
| `src/gpu_bundle.py` | préparation du bundle pour Colab |

Les sorties intermédiaires sont dans `outputs/regression/`.
