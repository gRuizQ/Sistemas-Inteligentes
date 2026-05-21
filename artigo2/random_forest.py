import os

import utils

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from algoritmo_cart import CARTDecisionTreeClassifier, CARTDecisionTreeRegressor


class RandomForestClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float = 2,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float, ensure_2d=True)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = self.classes_.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        for _ in range(int(self.n_estimators)):
            if self.bootstrap:
                indices = rng.integers(0, X.shape[0], size=X.shape[0], endpoint=False)
            else:
                indices = rng.choice(X.shape[0], size=X.shape[0], replace=False)

            tree_seed = int(rng.integers(0, 2**32 - 1, endpoint=True))
            tree = CARTDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree._fit_encoded(X[indices], y_encoded[indices], self.classes_)
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree in self.estimators_:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.estimators_)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RandomForestRegressorCustom(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[float] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float, ensure_2d=True)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        for _ in range(int(self.n_estimators)):
            if self.bootstrap:
                indices = rng.integers(0, X.shape[0], size=X.shape[0], endpoint=False)
            else:
                indices = rng.choice(X.shape[0], size=X.shape[0], replace=False)

            tree_seed = int(rng.integers(0, 2**32 - 1, endpoint=True))
            tree = CARTDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        preds = np.zeros(X.shape[0], dtype=float)
        for tree in self.estimators_:
            preds += tree.predict(X)
        return preds / len(self.estimators_)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(__file__)

    arquivo_treino = os.path.join(BASE_DIR, "02_treino_sinais_vitais_com_label.txt")
    X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler = utils.preparar_dados(
        arquivo_treino
    )

    rf_reg = RandomForestRegressorCustom(n_estimators=300, random_state=42)
    rf_clf = RandomForestClassifierCustom(n_estimators=300, random_state=42)

    modelo_reg_treinado = utils.treinar_e_avaliar_regressao(
        rf_reg, X_train_s, y_reg_train, X_test_s, y_reg_test, nome_modelo="RandomForest_Custom"
    )
    modelo_clf_treinado = utils.treinar_e_avaliar_classificacao(
        rf_clf, X_train_s, y_clf_train, X_test_s, y_clf_test, nome_modelo="RandomForest_Custom"
    )

    # 4. Geração do teste cego
    # Gerando as predições nos dados de teste para alimentar os gráficos
    predicoes_reg_teste = modelo_reg_treinado.predict(X_test_s)
    predicoes_clf_teste = modelo_clf_treinado.predict(X_test_s)

    utils.plotar_analise_regressao(y_reg_test, predicoes_reg_teste, nome_modelo="RandomForest_Custom Regressão")
    utils.plotar_analise_classificacao(y_clf_test, predicoes_clf_teste, nome_modelo="RandomForest_Custom Classificação")

    arquivo_cego = os.path.join(BASE_DIR, "01_treino_sinais_vitais_sem_label.txt")
    arquivo_saida = os.path.join(BASE_DIR, "resultados_predicao.csv")

    utils.gerar_arquivo_teste_cego(
        filepath_entrada=arquivo_cego,
        filepath_saida=arquivo_saida,
        scaler=scaler,
        modelo_reg=modelo_reg_treinado,
        modelo_clf=modelo_clf_treinado,
    )
