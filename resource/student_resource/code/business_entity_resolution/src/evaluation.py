import numpy as np
from typing import Dict, Set, List, Tuple, Any
from utils import calculate_entity_macro_f05, compute_f05

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    all_s1_ids: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth using exact challenge evaluation criteria.
    """
    if all_s1_ids is not None:
        # Guarantee all S1 entities are in maps
        for s1_id in all_s1_ids:
            if s1_id not in ground_truth:
                ground_truth[s1_id] = set()
            if s1_id not in predictions:
                predictions[s1_id] = set()
                
    macro_f05, macro_prec, macro_rec, singleton_acc = calculate_entity_macro_f05(ground_truth, predictions)
    
    # Calculate pair-level metrics
    total_true_pairs = sum(len(v) for v in ground_truth.values())
    total_pred_pairs = sum(len(v) for v in predictions.values())
    total_tp = sum(len(predictions.get(k, set()).intersection(ground_truth.get(k, set()))) for k in ground_truth)
    
    pair_precision = total_tp / total_pred_pairs if total_pred_pairs > 0 else 0.0
    pair_recall = total_tp / total_true_pairs if total_true_pairs > 0 else 0.0
    pair_f05 = compute_f05(pair_precision, pair_recall)
    
    # False merges & False negatives
    false_merges = total_pred_pairs - total_tp
    false_negatives = total_true_pairs - total_tp
    
    # Singletons
    singletons = [k for k, v in ground_truth.items() if len(v) == 0]
    singleton_correct = sum(1 for k in singletons if len(predictions.get(k, set())) == 0)
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "singleton_accuracy": singleton_acc,
        "singleton_count": len(singletons),
        "singleton_correct": singleton_correct,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f05": pair_f05,
        "total_true_pairs": total_true_pairs,
        "total_predicted_pairs": total_pred_pairs,
        "true_positives": total_tp,
        "false_merges": false_merges,
        "false_negatives": false_negatives
    }
