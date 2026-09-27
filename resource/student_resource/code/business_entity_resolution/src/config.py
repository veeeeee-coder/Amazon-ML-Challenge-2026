import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Random seed
    seed: int = 42
    
    # Paths
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    dataset_dir: str = field(default="")
    output_dir: str = field(default="")
    reports_dir: str = field(default="")
    models_dir: str = field(default="")
    
    # Blocking settings
    blocking_top_k: int = 50
    max_key_freq: int = 15000
    
    # Training settings
    train_sample_size: int = 15000
    val_split_ratio: float = 0.2
    
    # Model parameters (LightGBM)
    lgb_params: dict = field(default_factory=lambda: {
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
    })
    
    # Threshold optimization
    threshold_search_grid: List[float] = field(default_factory=lambda: [
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
    ])
    default_threshold: float = 0.55
    
    def __post_init__(self):
        if not self.dataset_dir:
            self.dataset_dir = os.path.join(self.base_dir, "dataset")
        if not self.output_dir:
            self.output_dir = os.path.join(self.base_dir, "output")
        if not self.reports_dir:
            self.reports_dir = os.path.join(self.base_dir, "reports")
        if not self.models_dir:
            self.models_dir = os.path.join(self.base_dir, "models")
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

DEFAULT_CONFIG = Config()
