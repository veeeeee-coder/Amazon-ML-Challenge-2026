import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List
from feature_engineering import FEATURE_NAMES
from utils import logger

class ModelTrainer:
    """Trains Gradient Boosted Pair Classification Model using LightGBM (MIT/Apache 2.0 compatible)."""
    def __init__(self, params: dict = None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 300
        }
        self.model = None
        self.feature_importances = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None
    ) -> lgb.Booster:
        logger.info(f"Training LightGBM on {len(X_train)} samples (Pos: {np.sum(y_train)}, Neg: {len(y_train)-np.sum(y_train)})")
        
        if feature_names is None:
            if X_train.shape[1] == len(FEATURE_NAMES):
                feature_names = FEATURE_NAMES
            else:
                feature_names = [f"f_{i}" for i in range(X_train.shape[1])]
                
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
            valid_sets.append(val_data)
            
        num_rounds = self.params.get('n_estimators', 300)
        params_copy = {k: v for k, v in self.params.items() if k != 'n_estimators'}
        
        self.model = lgb.train(
            params_copy,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=valid_sets
        )
        
        # Calculate feature importances
        raw_imp = self.model.feature_importance(importance_type='gain')
        for name, imp in zip(feature_names, raw_imp):
            self.feature_importances[name] = float(imp)
            
        # Log top 5 features
        sorted_imp = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top 5 predictive features (gain): {sorted_imp}")
        
        return self.model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained.")
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X)
