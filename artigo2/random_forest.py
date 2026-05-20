from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from algoritmo_cart import CARTDecisionTreeClassifier, CARTDecisionTreeRegressor


class RandomForestClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
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
        n_estimators: int = 100,
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
