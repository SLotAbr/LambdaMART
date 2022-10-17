import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class LambdaMART:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        pass

    def _get_data(self) -> List[np.ndarray]:
        pass

    def _prepare_data(self) -> None:
        pass

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        pass

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        pass

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        pass

    def fit(self):
        pass

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def compute_labels_in_batch(self, y_true):
        pass

    def compute_gain_diff(self, y_true, gain_scheme="exp2"):
        pass

    def _compute_lambdas(self, y_true: torch.FloatTensor, 
                               y_pred: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass
