"""Catalogue des modèles de régression : espace de recherche Optuna,
construction du pipeline (pré-traitement + estimateur) et paramètres fixés."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import optuna
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from src.features.preprocessing import (
    CATBOOST_CAT, CATBOOST_NUM, KNN_GROUPS, LGBM_CAT_IDX, LGBMG_CAT_IDX, SEED, knn_preprocessor,
    lgbm_group_preprocessor, lgbm_preprocessor, linear_preprocessor, poly_preprocessor,
    tree_preprocessor,
)


@dataclass
class ModelSpec:
    name: str
    label: str
    space: Callable[[optuna.Trial], dict]
    build: Callable[[dict], object]
    n_trials: int
    fixed: dict = field(default_factory=dict)   # paramètres non optimisés (pour le rapport)


def scaled_target(estimator):
    """Cible standardisée : l'epsilon du SVR s'exprime alors en écarts-types de y."""
    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


# ----------------------------------------------------------------- régression linéaire
def linear_space(t):
    return dict(alpha=t.suggest_float("alpha", 1e-4, 1e3, log=True),
                min_frequency=t.suggest_int("min_frequency", 5, 500, log=True))


def linear_build(p):
    return make_pipeline(linear_preprocessor(p["min_frequency"]), Ridge(alpha=p["alpha"]))


# ----------------------------------------------------------------- régression polynomiale
def poly_space(t):
    # bornes élargies : la recherche précédente retenait systématiquement degree=3 et
    # min_frequency=5 (bord de l'espace)
    return dict(degree=t.suggest_int("degree", 2, 4),
                interaction_only=t.suggest_categorical("interaction_only", [False, True]),
                alpha=t.suggest_float("alpha", 1e-3, 1e3, log=True),
                min_frequency=t.suggest_int("min_frequency", 2, 500, log=True))


def poly_build(p):
    return make_pipeline(poly_preprocessor(p["degree"], p["interaction_only"], p["min_frequency"]),
                         Ridge(alpha=p["alpha"]))


# ----------------------------------------------------------------- k plus proches voisins
def knn_space(t):
    # bornes élargies : deux poids de groupe atteignaient systématiquement 5 (bord de l'espace)
    p = {f"w_{g}": t.suggest_float(f"w_{g}", 0.0, 8.0) for g in KNN_GROUPS}
    p["min_frequency"] = t.suggest_int("min_frequency", 20, 500, log=True)
    p["n_neighbors"] = t.suggest_int("n_neighbors", 5, 150, log=True)
    p["weights"] = t.suggest_categorical("weights", ["uniform", "distance"])
    return p


def knn_build(p):
    gw = {g: p[f"w_{g}"] for g in KNN_GROUPS}
    return make_pipeline(knn_preprocessor(gw, p["min_frequency"]),
                         KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"],
                                             n_jobs=-1))


# ----------------------------------------------------------------- SVR à noyau gaussien
def svr_space(t):
    return dict(gamma=t.suggest_float("gamma", 1e-4, 1e-1, log=True),
                n_components=t.suggest_int("n_components", 100, 800, log=True),
                C=t.suggest_float("C", 1e-3, 1e2, log=True),
                epsilon=t.suggest_float("epsilon", 0.0, 0.5),
                min_frequency=t.suggest_int("min_frequency", 20, 500, log=True))


def svr_build(p):
    return make_pipeline(
        linear_preprocessor(p["min_frequency"]),
        Nystroem(kernel="rbf", gamma=p["gamma"], n_components=p["n_components"], random_state=SEED),
        scaled_target(LinearSVR(C=p["C"], epsilon=p["epsilon"], loss="squared_epsilon_insensitive",
                                dual=False, max_iter=5000, random_state=SEED)),
    )


# ----------------------------------------------------------------- arbre de décision (CART)
def cart_space(t):
    return dict(max_depth=t.suggest_int("max_depth", 3, 30),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 300, log=True),
                min_samples_split=t.suggest_int("min_samples_split", 2, 300, log=True),
                max_features=t.suggest_float("max_features", 0.3, 1.0),
                ccp_alpha=t.suggest_float("ccp_alpha", 1e-3, 1e3, log=True))


def cart_build(p):
    return make_pipeline(tree_preprocessor(), DecisionTreeRegressor(random_state=SEED, **p))


# ----------------------------------------------------------------- forêt aléatoire
def rf_space(t):
    # borne élargie : n_estimators atteignait systématiquement 400 (bord de l'espace)
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 800, step=50),
                max_depth=t.suggest_int("max_depth", 8, 40),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50, log=True),
                max_features=t.suggest_float("max_features", 0.2, 1.0),
                max_samples=t.suggest_float("max_samples", 0.3, 1.0))


def rf_build(p):
    return make_pipeline(tree_preprocessor(),
                         RandomForestRegressor(n_jobs=-1, random_state=SEED, **p))


# ----------------------------------------------------------------- gradient boosting
class EarlyStoppingLGBM(BaseEstimator, RegressorMixin):
    """LGBMRegressor dont le nombre d'arbres n'est PAS un hyperparamètre Optuna : un plafond
    élevé est fixé (`max_estimators`) et une fraction des données d'apprentissage est mise de
    côté à chaque `fit` pour l'arrêt anticipé, qui détermine le nombre d'arbres réellement
    utile. Cela libère le budget Optuna pour les hyperparamètres qui comptent vraiment (plutôt
    que de le dépenser à redécouvrir indirectement "combien d'arbres" à chaque essai)."""

    def __init__(self, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=1.0,
                colsample_bytree=1.0, reg_lambda=0.0, reg_alpha=0.0, cat_smooth=10.0,
                cat_l2=10.0, min_data_per_group=100, categorical_feature=None,
                max_estimators=5000, early_stopping_rounds=100, val_fraction=0.1,
                random_state=SEED):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.min_data_per_group = min_data_per_group
        self.categorical_feature = categorical_feature
        self.max_estimators = max_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, random_state=self.random_state)
        self.model_ = LGBMRegressor(
            n_estimators=self.max_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            subsample=self.subsample, subsample_freq=1, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, cat_smooth=self.cat_smooth,
            cat_l2=self.cat_l2, min_data_per_group=self.min_data_per_group,
            categorical_feature=self.categorical_feature, random_state=self.random_state,
            n_jobs=-1, verbose=-1)
        self.model_.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                        callbacks=[early_stopping(self.early_stopping_rounds, verbose=False)])
        self.best_iteration_ = self.model_.best_iteration_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def lgbm_space(t):
    return dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                min_child_samples=t.suggest_int("min_child_samples", 5, 100, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
                reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                cat_smooth=t.suggest_float("cat_smooth", 1.0, 100.0, log=True),
                cat_l2=t.suggest_float("cat_l2", 1.0, 100.0, log=True),
                min_data_per_group=t.suggest_int("min_data_per_group", 10, 200, log=True))


def lgbm_build(p):
    return make_pipeline(lgbm_preprocessor(), EarlyStoppingLGBM(categorical_feature=LGBM_CAT_IDX, **p))


# ------------------------------------------------- gradient boosting + variables de groupe
def lgbmg_space(t):
    # pas d'apprentissage plus fins que pour `lgbm` : l'arrêt anticipé ajoute les arbres
    # nécessaires sans coût de réglage, au prix d'un temps de calcul plus long
    return dict(learning_rate=t.suggest_float("learning_rate", 0.004, 0.05, log=True),
                num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
                reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                cat_smooth=t.suggest_float("cat_smooth", 1.0, 100.0, log=True),
                cat_l2=t.suggest_float("cat_l2", 1.0, 100.0, log=True),
                min_data_per_group=t.suggest_int("min_data_per_group", 10, 200, log=True))


def lgbmg_build(p):
    return make_pipeline(lgbm_group_preprocessor(),
                         EarlyStoppingLGBM(categorical_feature=LGBMG_CAT_IDX, max_estimators=20000, **p))


class EarlyStoppingCatBoost(BaseEstimator, RegressorMixin):
    """CatBoost avec ses catégorielles natives (statistiques ordonnées sur la cible, calculées
    sur des permutations aléatoires : chaque ligne n'est encodée qu'avec les lignes qui la
    précèdent) et combinaisons automatiques de catégorielles.
    Nombre d'arbres fixé par arrêt anticipé sur 10 % des données d'apprentissage."""

    def __init__(self, depth=6, learning_rate=0.06, l2_leaf_reg=3.0, random_strength=1.0,
                 bagging_temperature=1.0, max_ctr_complexity=4, max_iterations=20000,
                 early_stopping_rounds=200, val_fraction=0.1, random_state=SEED, thread_count=-1):
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.max_ctr_complexity = max_ctr_complexity
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.thread_count = thread_count

    def _frame(self, X):
        F = X[CATBOOST_NUM + CATBOOST_CAT].copy()
        F[CATBOOST_CAT] = F[CATBOOST_CAT].astype(str)
        return F

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        F = self._frame(X)
        F_tr, F_val, y_tr, y_val = train_test_split(F, y, test_size=self.val_fraction,
                                                    random_state=self.random_state)
        self.model_ = CatBoostRegressor(
            loss_function="RMSE", iterations=self.max_iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength, bagging_temperature=self.bagging_temperature,
            max_ctr_complexity=self.max_ctr_complexity, cat_features=CATBOOST_CAT,
            od_type="Iter", od_wait=self.early_stopping_rounds, random_seed=self.random_state,
            thread_count=self.thread_count, verbose=False, allow_writing_files=False)
        self.model_.fit(F_tr, y_tr, eval_set=(F_val, y_val), use_best_model=True)
        self.best_iteration_ = self.model_.get_best_iteration()
        return self

    def predict(self, X):
        return self.model_.predict(self._frame(X))


def catg_space(t):
    return dict(depth=t.suggest_int("depth", 5, 9),
                learning_rate=t.suggest_float("learning_rate", 0.04, 0.15, log=True),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
                random_strength=t.suggest_float("random_strength", 0.1, 5.0, log=True),
                bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 2.0),
                max_ctr_complexity=t.suggest_int("max_ctr_complexity", 1, 4))


def catg_build(p):
    return EarlyStoppingCatBoost(**p)


# ------------------------------------------ cible normalisée par le poids de corps
class BodyweightRatioTarget(BaseEstimator, RegressorMixin):
    """Apprend z = TotalKg / BW^p puis prédit z_hat * BW^p (p = `power`).

    La force croît à peu près comme une puissance du poids de corps (loi allométrique) : la
    cible normalisée varie moins d'un individu à l'autre et se prête mieux aux coupures des
    arbres ; les encodages par la cible des clés de groupe deviennent un « niveau de force
    relatif au poids », comparable d'un athlète à l'autre. La perte quadratique en z pondère
    les erreurs en kg par BW^(-2p)."""

    def __init__(self, estimator, power=0.8):
        self.estimator = estimator
        self.power = power

    def fit(self, X, y):
        from sklearn.base import clone
        scale = X["BodyweightKg"].to_numpy(float) ** self.power
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, float) / scale)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) * X["BodyweightKg"].to_numpy(float) ** self.power


def lgbmgr_space(t):
    return dict(lgbmg_space(t), bw_power=t.suggest_float("bw_power", 0.4, 1.2))


def lgbmgr_build(p):
    p = dict(p)
    power = p.pop("bw_power")
    return BodyweightRatioTarget(lgbmg_build(p), power)


def catgr_space(t):
    return dict(catg_space(t), bw_power=t.suggest_float("bw_power", 0.4, 1.2))


def catgr_build(p):
    p = dict(p)
    power = p.pop("bw_power")
    return BodyweightRatioTarget(catg_build(p), power)


MODELS = {
    "linear": ModelSpec("linear", "Régression linéaire (ridge)", linear_space, linear_build, 40,
                        dict(fit_intercept=True, solver="auto", tol=1e-4)),
    "poly": ModelSpec("poly", "Régression polynomiale (ridge)", poly_space, poly_build, 35,
                      dict(include_bias=False, solver="auto")),
    "knn": ModelSpec("knn", "k plus proches voisins", knn_space, knn_build, 35,
                     dict(p=2, metric="minkowski", algorithm="auto", leaf_size=30)),
    "svr": ModelSpec("svr", "SVR à noyau gaussien (Nyström)", svr_space, svr_build, 25,
                     dict(kernel="rbf", loss="squared_epsilon_insensitive", dual=False,
                          max_iter=5000, tol=1e-4, fit_intercept=True)),
    "cart": ModelSpec("cart", "Arbre de décision (CART)", cart_space, cart_build, 50,
                      dict(criterion="squared_error", splitter="best", max_leaf_nodes=None,
                           min_impurity_decrease=0.0)),
    "rf": ModelSpec("rf", "Forêt aléatoire", rf_space, rf_build, 30,
                    dict(criterion="squared_error", bootstrap=True, min_samples_split=2,
                         ccp_alpha=0.0, max_leaf_nodes=None)),
    "lgbm": ModelSpec("lgbm", "Gradient boosting (LightGBM)", lgbm_space, lgbm_build, 40,
                      dict(boosting_type="gbdt", max_depth=-1, subsample_freq=1, max_bin=255,
                           min_split_gain=0.0, min_child_weight=1e-3, max_cat_to_onehot=4,
                           max_estimators=5000, early_stopping_rounds=100, val_fraction=0.1)),
    # boostings « structurés » : variables de groupe encodées par la cible, cible y / BW^p
    "lgbmgr": ModelSpec("lgbmgr", "LightGBM + groupes, cible / BW^p", lgbmgr_space, lgbmgr_build, 40,
                        dict(boosting_type="gbdt", max_estimators=20000, early_stopping_rounds=100,
                             val_fraction=0.1)),
    "catgr": ModelSpec("catgr", "CatBoost + groupes, cible / BW^p", catgr_space, catgr_build, 10,
                       dict(loss_function="RMSE", max_iterations=20000, od_wait=200, val_fraction=0.1)),
}
