import numpy as np
from typing import Dict, Set, List, Tuple
from collections import defaultdict
from utils import calculate_entity_macro_f05, logger

class ThresholdOptimizer:
    """
    Optimizes decision threshold specifically for Macro F0.5 metric.
    Searches across grid to find threshold maximizing Macro F0.5.
    """
    def __init__(self, search_grid: List[float] = None):
        self.search_grid = search_grid or [
            0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        ]
        self.best_threshold = 0.55
        self.best_score = 0.0
        self.threshold_history = []

    def optimize(
        self,
        val_pair_ids: List[Tuple[str, str]],
        val_probabilities: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        val_s1_ids: List[str]
    ) -> float:
        """
        Evaluates Macro F0.5 for each threshold and returns optimal threshold.
        """
        # Map s1_id -> list of (cand_id, prob)
        pair_dict = defaultdict(list)
        for idx, (s1_id, cand_id) in enumerate(val_pair_ids):
            pair_dict[s1_id].append((cand_id, val_probabilities[idx]))
            
        best_t = self.search_grid[0]
        best_f05 = -1.0
        self.threshold_history = []
        
        for t in self.search_grid:
            pred_map = {}
            for s1_id in val_s1_ids:
                matches = set()
                for cand_id, prob in pair_dict.get(s1_id, []):
                    if prob >= t:
                        matches.add(cand_id)
                pred_map[s1_id] = matches
                
            macro_f05, macro_prec, macro_rec, s_acc = calculate_entity_macro_f05(ground_truth, pred_map)
            self.threshold_history.append({
                "threshold": t,
                "macro_f05": macro_f05,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "singleton_accuracy": s_acc
            })
            
            logger.info(f"Threshold: {t:.2f} -> Macro F0.5: {macro_f05:.4f} (Prec: {macro_prec:.4f}, Rec: {macro_rec:.4f}, Singleton Acc: {s_acc:.4f})")
            
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_t = t
                
        self.best_threshold = best_t
        self.best_score = best_f05
        logger.info(f"Optimal Decision Threshold: {self.best_threshold:.2f} with Validation Macro F0.5: {self.best_score:.4f}")
        return self.best_threshold
