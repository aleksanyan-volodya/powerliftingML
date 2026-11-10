"""Chargement, séparation A/B et pré-traitements par famille de modèles.

Principe : on n'ajoute que des variables qui *extraient une information* cachée dans
du texte (valeur numérique d'une catégorie de poids, milieu d'une tranche d'âge...).
Les non-linéarités et interactions sont laissées aux modèles (polynôme, noyau,
arbres) ; leur réglage est fait par Optuna.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler, TargetEncoder,
)

TARGET = "TotalKg"
SEED = 42

NUM_BASE = ["Age", "BodyweightKg", "MeetYear"]
NUM_DERIVED = [
    "LogBW",                                                   # loi allométrique Total ~ BW^(2/3)
    "WeightClassNum", "WeightClassIsPlus", "WeightClassUnknown", "BWMinusClass",
    "AgeClassMid", "AgeClassUnknown",
    "BirthYearClassOrd", "BirthYearClassUnknown",
    "AgeIsHalf",                                               # âge approximatif (x.5)
    "CompetesAbroad",                                          # compétition internationale
]
NUMERIC = NUM_BASE + NUM_DERIVED
KEY_CAT = ["Sex", "Equipment", "Tested"]
OTHER_CAT = ["Country", "State", "Federation", "ParentFederation",
             "MeetCountry", "MeetState", "Sanctioned"]
NOMINAL = KEY_CAT + OTHER_CAT

ORDINAL_CAT = ["AgeClass", "BirthYearClass", "WeightClassKg"]

BIRTH_YEAR_ORDER = {"14-18": 0, "19-23": 1, "24-39": 2, "40-49": 3,
                    "50-59": 4, "60-69": 5, "70-999": 6}


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformation ligne à ligne (aucune statistique apprise -> pas de fuite)."""
    d = df.copy()
    d["LogBW"] = np.log(d["BodyweightKg"])

    wc = d["WeightClassKg"].astype(str)
    d["WeightClassIsPlus"] = wc.str.endswith("+").astype(float)
    num = pd.to_numeric(wc.str.rstrip("+"), errors="coerce").abs()   # '-93' mal saisi
    d["WeightClassUnknown"] = num.isna().astype(float)
    d["WeightClassNum"] = num.fillna(d["BodyweightKg"])
    d["BWMinusClass"] = np.where(d["WeightClassIsPlus"] == 1, 0.0,
                                 d["BodyweightKg"] - d["WeightClassNum"])

    bounds = d["AgeClass"].astype(str).str.extract(r"^(\d+)-(\d+)$").astype(float)
    bounds[1] = bounds[1].clip(upper=bounds[0] + 5)                  # '90-999'
    mid = bounds.mean(axis=1)
    d["AgeClassUnknown"] = mid.isna().astype(float)
    d["AgeClassMid"] = mid.fillna(d["Age"])

    by = d["BirthYearClass"].map(BIRTH_YEAR_ORDER)
    d["BirthYearClassUnknown"] = by.isna().astype(float)
    d["BirthYearClassOrd"] = by.fillna(2)

    d["AgeIsHalf"] = (d["Age"] % 1 == 0.5).astype(float)
    d["CompetesAbroad"] = ((d["Country"] != d["MeetCountry"])
                           & (d["Country"] != "Unknown")).astype(float)
    out = d[NUMERIC + NOMINAL + ORDINAL_CAT].copy()
    out["BirthYear"] = d["MeetYear"] - d["Age"]                       # année de naissance estimée
    d["BirthYearR"] = out["BirthYear"].round(0).astype(int)
    for name, cols in GROUP_KEY_DEF.items():
        parts = [d[c].astype(str) for c in cols]
        out[name] = parts[0].str.cat(parts[1:], sep="|")
    return out


# ---------------------------------------------------------------- variables de groupe
# Clés de groupe (chaînes), encodées par la cible À l'intérieur des pipelines (validation
# croisée interne donc pas de fuite) : la rencontre (98 % des lignes de test appartiennent à une
# rencontre vue au train), un « profil d'athlète » (sexe, pays, état, année de naissance) et
# les « jumeaux » (mêmes valeurs sur toutes les variables : même athlète inscrit dans plusieurs
# divisions d'une même rencontre, total le plus souvent identique).
RAW_VARS = ["Age", "BodyweightKg", "MeetYear", "Sex", "Equipment", "AgeClass", "BirthYearClass",
            "WeightClassKg", "Tested", "Country", "State", "Federation", "ParentFederation",
            "MeetCountry", "MeetState", "Sanctioned"]
GROUP_KEY_DEF = {
    "meet": ["Federation", "MeetCountry", "MeetState", "MeetYear"],
    "meet_sex_eq": ["Federation", "MeetCountry", "MeetState", "MeetYear", "Sex", "Equipment"],
    "ath": ["Sex", "Country", "State", "BirthYearR"],
    "ath_eq": ["Sex", "Country", "State", "BirthYearR", "Equipment"],
    "ath_fed": ["Sex", "Country", "State", "BirthYearR", "Federation"],
    "ath_wc": ["Sex", "Country", "State", "BirthYearR", "WeightClassKg"],
    "twin": RAW_VARS,
    "twin2": [c for c in RAW_VARS if c not in ("AgeClass", "BirthYearClass", "WeightClassKg", "Tested")],
}
GROUP_KEYS = list(GROUP_KEY_DEF)
GROUP_NUM = ["BirthYear"] + [f"n_{k}" for k in GROUP_KEYS]


def add_group_counts(frames):
    """Effectif de chaque groupe, compté sur l'ensemble des lignes (train + test) SANS la
    cible"""
    allk = pd.concat([f[GROUP_KEYS] for f in frames], ignore_index=True)
    vcs = {k: allk[k].value_counts() for k in GROUP_KEYS}
    out = []
    for f in frames:
        f = f.copy()
        for k in GROUP_KEYS:
            f[f"n_{k}"] = f[k].map(vcs[k]).astype(float)
        out.append(f)
    return out


def load_locked(raw_dir: str = "data/raw", test_size: float = 0.2):
    """Sépare le train en D (développement, 80 %) et T (test 20 %).

    TOUTE la sélection (hyperparamètres par validation croisée, prédictions hors bloc,
    pondération de l'ensemble, AutoML, contexte des modèles de fondation) se fait sur D.
    T n'est consulté qu'une fois, à la fin, pour estimer sans biais l'erreur de chaque
    procédure.

    Retourne X_D, y_D, X_T, y_T, X_test, test_ids."""

    X, y, X_test, test_ids = load_full(raw_dir)
    idx = np.random.default_rng(SEED).permutation(len(y))
    n_T = int(round(test_size * len(y)))
    T, D = np.sort(idx[:n_T]), np.sort(idx[n_T:])
    return (X.iloc[D].reset_index(drop=True), y.iloc[D].reset_index(drop=True),
            X.iloc[T].reset_index(drop=True), y.iloc[T].reset_index(drop=True), X_test, test_ids)


def load_full(raw_dir: str = "data/raw"):
    """Retourne X, y, X_test, test_ids (toutes les lignes étiquetées). Voir load_locked pour
    le découpage développement / test utilisé par le pipeline."""
    train = pd.read_csv(f"{raw_dir}/train.csv")
    test = pd.read_csv(f"{raw_dir}/test.csv")
    X, X_test = add_group_counts([add_base_features(train.drop(columns=[TARGET, "id"])),
                                  add_base_features(test.drop(columns=["id"]))])
    return X, train[TARGET], X_test, test["id"]


# ----------------------------------------------------------------------------- encodeurs
def onehot(min_frequency: int) -> OneHotEncoder:
    """One-hot des nominales ; les modalités vues moins de `min_frequency` fois sont
    regroupées dans une modalité 'infrequent', qui accueille aussi les modalités inconnues."""
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                         sparse_output=False)


def linear_preprocessor(min_frequency: int) -> ColumnTransformer:
    """Régression linéaire, SVR : numériques standardisées + one-hot des nominales ET des
    catégorielles ordinales à forte cardinalité (voir ORDINAL_CAT) : ces dernières gardent
    ainsi un coefficient par modalité, en plus de leur résumé numérique (dans NUMERIC)."""
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", onehot(min_frequency), NOMINAL + ORDINAL_CAT)])


def poly_preprocessor(degree: int, interaction_only: bool, min_frequency: int) -> ColumnTransformer:
    """Régression polynomiale : base polynomiale de degré `degree` sur les variables
    continues principales ET les indicatrices Sex/Equipment/Tested (les interactions
    sexe x poids, équipement x âge... sont ainsi générées automatiquement)."""
    poly_block = Pipeline([
        ("enc", ColumnTransformer([
            ("num", StandardScaler(), NUM_BASE),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), KEY_CAT),
        ])),
        ("poly", PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                                    include_bias=False)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("poly", poly_block, NUM_BASE + KEY_CAT),
        ("num", StandardScaler(), NUM_DERIVED),
        ("cat", onehot(min_frequency), OTHER_CAT + ORDINAL_CAT),
    ])


KNN_GROUPS = {
    "bw": (["BodyweightKg", "LogBW", "WeightClassNum", "BWMinusClass"], "num"),
    "age": (["Age", "AgeClassMid", "BirthYearClassOrd"], "num"),
    "year": (["MeetYear"], "num"),
    "flags": (["WeightClassIsPlus", "WeightClassUnknown", "AgeClassUnknown",
               "BirthYearClassUnknown", "AgeIsHalf", "CompetesAbroad"], "num"),
    "key_cat": (KEY_CAT, "cat"),
    "other_cat": (OTHER_CAT, "cat"),
}


def knn_preprocessor(group_weights: dict, min_frequency: int) -> ColumnTransformer:
    """k-NN : la distance dépend de l'échelle -> standardisation + one-hot, puis un poids
    par groupe de variables (appris par Optuna ; poids nul = groupe supprimé)."""
    transformers, weights = [], {}
    for name, (cols, kind) in KNN_GROUPS.items():
        if group_weights[name] <= 0:
            continue
        enc = StandardScaler() if kind == "num" else onehot(min_frequency)
        transformers.append((name, enc, cols))
        weights[name] = group_weights[name]
    return ColumnTransformer(transformers, transformer_weights=weights)


def tree_preprocessor(smooth="auto") -> ColumnTransformer:
    """CART et forêt aléatoire : pas de standardisation (les arbres sont invariants par
    transformation monotone). Les nominales sont remplacées par la moyenne (lissée) de la
    cible dans la modalité (target encoding avec validation croisée interne -> pas de fuite) :
    un ordre pertinent pour les coupures, et 1 colonne par variable au lieu de centaines."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", TargetEncoder(target_type="continuous", smooth=smooth, random_state=SEED), NOMINAL),
    ])


def lgbm_preprocessor() -> ColumnTransformer:
    """LightGBM : codes entiers pour les nominales, traitées ensuite nativement comme
    catégorielles (partition optimale des modalités) -> ni one-hot ni target encoding."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), NOMINAL),
    ])


LGBM_CAT_IDX = list(range(len(NUMERIC), len(NUMERIC) + len(NOMINAL)))


def lgbm_group_preprocessor() -> ColumnTransformer:
    """LightGBM avec variables de groupe : numériques + effectifs bruts, nominales en codes
    entiers (catégorielles natives), clés de groupe encodées par la cible (moyenne lissée,
    validation croisée interne à 5 blocs : la cible d'une ligne n'entre jamais dans son propre
    encodage)."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC + GROUP_NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), NOMINAL),
        ("te", TargetEncoder(target_type="continuous", random_state=SEED), GROUP_KEYS),
    ])


LGBMG_CAT_IDX = list(range(len(NUMERIC + GROUP_NUM), len(NUMERIC + GROUP_NUM) + len(NOMINAL)))
CATBOOST_NUM = NUMERIC + GROUP_NUM
CATBOOST_CAT = NOMINAL + ORDINAL_CAT + GROUP_KEYS
