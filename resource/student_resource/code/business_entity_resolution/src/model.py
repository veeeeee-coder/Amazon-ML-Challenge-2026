import numpy as np
import lightgbm as lgb
from feature_engineering import FEATURE_NAMES

class EntityResolutionModel:
    """LightGBM Pairwise Entity Resolution Model with F0.5 metric optimization."""
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
            valid_sets.append(val_data)
            
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=valid_sets
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X)

    def predict_matches(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return probas >= self.threshold
