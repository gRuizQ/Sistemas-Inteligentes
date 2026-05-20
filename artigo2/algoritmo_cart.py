import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class _Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def _gini(y_encoded: np.ndarray, n_classes: int) -> float:
    if y_encoded.size == 0:
        return 0.0
    counts = np.bincount(y_encoded, minlength=n_classes).astype(float)
    probs = counts / counts.sum()
    return 1.0 - float(np.sum(probs * probs))


def _mse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    mean = float(np.mean(y))
    diff = y - mean
    return float(np.mean(diff * diff))


def _resolve_max_features(max_features: Optional[int | float], n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        key = max_features.lower()
        if key == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if key == "log2":
            return max(1, int(math.log2(n_features)))
        raise ValueError("max_features string deve ser 'sqrt' ou 'log2'.")
    if isinstance(max_features, int):
        if max_features <= 0 or max_features > n_features:
            raise ValueError("max_features int deve estar em [1, n_features].")
        return max_features
    if isinstance(max_features, float):
        if max_features <= 0.0 or max_features > 1.0:
            raise ValueError("max_features float deve estar em (0, 1].")
        return max(1, int(math.ceil(max_features * n_features)))
    raise ValueError("max_features inválido.")


def _candidate_thresholds(feature_values: np.ndarray, max_candidates: int = 64) -> np.ndarray:
    unique_vals = np.unique(feature_values)
    if unique_vals.size <= 1:
        return np.array([], dtype=float)
    mids = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    if mids.size <= max_candidates:
        return mids
    idx = np.linspace(0, mids.size - 1, num=max_candidates, dtype=int)
    return mids[idx]


class CARTDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float, ensure_2d=True)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = self.classes_.shape[0]
        self._rng_ = np.random.default_rng(self.random_state)
        self._max_features_ = _resolve_max_features(self.max_features, self.n_features_in_)
        self.tree_ = self._build_tree(X, y_encoded, depth=0)
        return self

    def _fit_encoded(self, X, y_encoded, classes_):
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        y_encoded = np.asarray(y_encoded, dtype=int)
        self.classes_ = np.asarray(classes_)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = self.classes_.shape[0]
        self._rng_ = np.random.default_rng(self.random_state)
        self._max_features_ = _resolve_max_features(self.max_features, self.n_features_in_)
        self.tree_ = self._build_tree(X, y_encoded, depth=0)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        probs = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i in range(X.shape[0]):
            node = self.tree_
            while not node.is_leaf:
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            probs[i, :] = node.value
        return probs

    def _build_tree(self, X: np.ndarray, y_encoded: np.ndarray, depth: int) -> _Node:
        n_samples = X.shape[0]
        counts = np.bincount(y_encoded, minlength=self.n_classes_).astype(float)
        proba = counts / counts.sum()

        if n_samples < self.min_samples_split:
            return _Node(value=proba)
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(value=proba)
        if np.count_nonzero(counts) == 1:
            return _Node(value=proba)

        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

        feature_candidates = self._rng_.choice(
            self.n_features_in_, size=self._max_features_, replace=False
        )
        base_impurity = _gini(y_encoded, self.n_classes_)

        for feature_index in feature_candidates:
            x_col = X[:, feature_index]
            order = np.argsort(x_col, kind="mergesort")
            x_sorted = x_col[order]
            if x_sorted[0] == x_sorted[-1]:
                continue
            y_sorted = y_encoded[order]

            n_left = np.arange(1, n_samples, dtype=int)
            n_right = n_samples - n_left
            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            valid &= x_sorted[:-1] != x_sorted[1:]
            if not np.any(valid):
                continue

            onehot = np.zeros((n_samples, self.n_classes_), dtype=float)
            onehot[np.arange(n_samples), y_sorted] = 1.0
            left_counts = np.cumsum(onehot, axis=0)[:-1]
            right_counts = counts - left_counts

            probs_left = left_counts / n_left[:, None]
            probs_right = right_counts / n_right[:, None]
            gini_left = 1.0 - np.sum(probs_left * probs_left, axis=1)
            gini_right = 1.0 - np.sum(probs_right * probs_right, axis=1)
            weighted = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

            weighted_valid = weighted[valid]
            best_pos_local = int(np.argmin(weighted_valid))
            split_index = int(np.flatnonzero(valid)[best_pos_local])
            best_impurity_local = float(weighted[split_index])

            if best_impurity_local < best_impurity:
                best_impurity = best_impurity_local
                best_feature = int(feature_index)
                best_threshold = float((x_sorted[split_index] + x_sorted[split_index + 1]) / 2.0)

        if best_feature is None:
            return _Node(value=proba)
        if base_impurity - best_impurity <= 1e-12:
            return _Node(value=proba)

        x_col = X[:, best_feature]
        left_mask = x_col <= best_threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y_encoded[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y_encoded[right_mask], depth + 1)
        return _Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)


class CARTDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int | float] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float, ensure_2d=True)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self._rng_ = np.random.default_rng(self.random_state)
        self._max_features_ = _resolve_max_features(self.max_features, self.n_features_in_)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        preds = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            node = self.tree_
            while not node.is_leaf:
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds[i] = float(node.value[0])
        return preds

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = X.shape[0]
        mean_value = float(np.mean(y))

        if n_samples < self.min_samples_split:
            return _Node(value=np.array([mean_value], dtype=float))
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(value=np.array([mean_value], dtype=float))
        if np.allclose(y, y[0]):
            return _Node(value=np.array([mean_value], dtype=float))

        best_feature = None
        best_threshold = None
        best_impurity = float("inf")
        base_impurity = _mse(y)

        feature_candidates = self._rng_.choice(
            self.n_features_in_, size=self._max_features_, replace=False
        )

        for feature_index in feature_candidates:
            x_col = X[:, feature_index]
            order = np.argsort(x_col, kind="mergesort")
            x_sorted = x_col[order]
            if x_sorted[0] == x_sorted[-1]:
                continue
            y_sorted = y[order]

            n_left = np.arange(1, n_samples, dtype=float)
            n_right = n_samples - n_left
            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            valid &= x_sorted[:-1] != x_sorted[1:]
            if not np.any(valid):
                continue

            y_cum = np.cumsum(y_sorted, dtype=float)
            y2_cum = np.cumsum(y_sorted * y_sorted, dtype=float)
            total_sum = float(y_cum[-1])
            total_sum_sq = float(y2_cum[-1])

            left_sum = y_cum[:-1]
            left_sum_sq = y2_cum[:-1]
            right_sum = total_sum - left_sum
            right_sum_sq = total_sum_sq - left_sum_sq

            mse_left = left_sum_sq / n_left - (left_sum / n_left) ** 2
            mse_right = right_sum_sq / n_right - (right_sum / n_right) ** 2
            weighted = (n_left / n_samples) * mse_left + (n_right / n_samples) * mse_right

            weighted_valid = weighted[valid]
            best_pos_local = int(np.argmin(weighted_valid))
            split_index = int(np.flatnonzero(valid)[best_pos_local])
            best_impurity_local = float(weighted[split_index])

            if best_impurity_local < best_impurity:
                best_impurity = best_impurity_local
                best_feature = int(feature_index)
                best_threshold = float((x_sorted[split_index] + x_sorted[split_index + 1]) / 2.0)

        if best_feature is None:
            return _Node(value=np.array([mean_value], dtype=float))
        if base_impurity - best_impurity <= 1e-12:
            return _Node(value=np.array([mean_value], dtype=float))

        x_col = X[:, best_feature]
        left_mask = x_col <= best_threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)
