import logging
import time
import json
import os
import sys
from typing import Set, Dict, List, Tuple
import numpy as np

def setup_logger(name: str = "EntityResolution", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configures structured Python logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

logger = setup_logger()

class Timer:
    """Context manager for profiling runtime execution."""
    def __init__(self, description: str):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Started: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        logger.info(f"Finished: {self.description} in {self.elapsed:.2f}s")

def compute_f05(precision: float, recall: float) -> float:
    """Calculates F0.5 score: F0.5 = (1.25 * P * R) / (0.25 * P + R)."""
    if precision + recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def calculate_entity_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Tuple[float, float, float, float]:
    """
    Computes Exact Challenge Metric: Macro-Averaged F0.5 across all Source 1 entities.
    
    Includes singleton scoring:
    - S1 entity with 0 true matches scores 1.0 if predicted empty, and 0.0 if false matches predicted.
    - S1 entity with true matches evaluated using F0.5 formula.
    
    Returns: (macro_f05, macro_precision, macro_recall, singleton_accuracy)
    """
    f05_scores = []
    prec_scores = []
    rec_scores = []
    singleton_scores = []
    
    all_s1_ids = set(ground_truth.keys()).union(predictions.keys())
    
    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        
        if not true_set:
            # Singleton entity
            if not pred_set:
                f05 = 1.0
                prec = 1.0
                rec = 1.0
                singleton_scores.append(1.0)
            else:
                f05 = 0.0
                prec = 0.0
                rec = 0.0
                singleton_scores.append(0.0)
        else:
            if not pred_set:
                f05 = 0.0
                prec = 0.0
                rec = 0.0
            else:
                tp = len(pred_set.intersection(true_set))
                prec = tp / len(pred_set) if pred_set else 0.0
                rec = tp / len(true_set) if true_set else 0.0
                f05 = compute_f05(prec, rec)
                
        f05_scores.append(f05)
        prec_scores.append(prec)
        rec_scores.append(rec)
        
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_prec = float(np.mean(prec_scores)) if prec_scores else 0.0
    macro_rec = float(np.mean(rec_scores)) if rec_scores else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0
    
    return macro_f05, macro_prec, macro_rec, singleton_acc

def save_json(data: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
