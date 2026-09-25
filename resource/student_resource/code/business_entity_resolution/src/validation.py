import numpy as np
import polars as pl
import pandas as pd
import json
import os
from typing import Dict, Set, List, Any
from collections import defaultdict

from utils import logger, save_json
from evaluation import evaluate_predictions
from train_model import ModelTrainer
from threshold_optimizer import ThresholdOptimizer
from matcher import EntityMatcher
from pair_dataset import build_pairwise_dataset
from feature_engineering import FEATURE_NAMES

def run_ablation_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    val_pair_ids: list,
    val_gt: dict,
    val_s1_ids: list,
    reports_dir: str
) -> pd.DataFrame:
    """
    Runs systematic ablation experiments:
    - Exp A: Name features only
    - Exp B: Address features only
    - Exp C: Name + Address
    - Exp D: Name + Address + Country
    - Exp E: Name + Address + Country + Blocking
    - Exp F: Full Model
    """
    logger.info("--- Running Ablation Experiments ---")
    
    # Feature index groups
    name_indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("name_")]
    addr_indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("addr_") or name.startswith("dig_")]
    joint_indices = [i for i, name in enumerate(FEATURE_NAMES) if name in ["comb_token_set", "country_exact_match", "is_s2"]]
    blocking_indices = [i for i, name in enumerate(FEATURE_NAMES) if name == "blocking_score"]
    
    experiments = [
        ("Exp_A_Name_Only", name_indices),
        ("Exp_B_Address_Only", addr_indices),
        ("Exp_C_Name_Address", name_indices + addr_indices),
        ("Exp_D_Name_Address_Country", name_indices + addr_indices + joint_indices),
        ("Exp_E_Name_Addr_Country_Blocking", name_indices + addr_indices + joint_indices + blocking_indices),
        ("Exp_F_Full_Model", list(range(len(FEATURE_NAMES))))
    ]
    
    results = []
    
    for exp_id, feat_cols in experiments:
        trainer = ModelTrainer()
        exp_feat_names = [FEATURE_NAMES[i] for i in feat_cols]
        trainer.train(X_train[:, feat_cols], y_train, feature_names=exp_feat_names)
        val_probs = trainer.predict_proba(X_val[:, feat_cols])
        
        # Optimize threshold
        opt = ThresholdOptimizer(search_grid=[0.30, 0.40, 0.50, 0.55, 0.60, 0.70])
        best_t = opt.optimize(val_pair_ids, val_probs, val_gt, val_s1_ids)
        
        matcher = EntityMatcher(threshold=best_t)
        preds = matcher.match_candidates(val_pair_ids, val_probs, val_s1_ids)
        
        eval_metrics = evaluate_predictions(val_gt, {k: set(v) for k, v in preds.items()}, val_s1_ids)
        
        results.append({
            "Experiment_ID": exp_id,
            "Num_Features": len(feat_cols),
            "Optimal_Threshold": best_t,
            "Macro_F05": eval_metrics["macro_f05"],
            "Macro_Precision": eval_metrics["macro_precision"],
            "Macro_Recall": eval_metrics["macro_recall"],
            "Singleton_Accuracy": eval_metrics["singleton_accuracy"],
            "Pair_Precision": eval_metrics["pair_precision"],
            "Pair_Recall": eval_metrics["pair_recall"]
        })
        
    df_exp = pd.DataFrame(results)
    exp_csv_path = os.path.join(reports_dir, "experiment_results.csv")
    df_exp.to_csv(exp_csv_path, index=False)
    logger.info(f"Saved ablation study results to {exp_csv_path}")
    return df_exp

def run_error_analysis(
    val_pair_ids: list,
    val_probs: np.ndarray,
    val_gt: dict,
    val_preds: dict,
    df_s1: pl.DataFrame,
    df_s2: pl.DataFrame,
    df_s3: pl.DataFrame,
    reports_dir: str
) -> dict:
    """Generates comprehensive error analysis with sample true positives, false positives, and false negatives."""
    s1_dict = {r["entity_id"]: r for r in df_s1.to_dicts()}
    s23_dict = {r["entity_id"]: r for r in pl.concat([df_s2, df_s3]).to_dicts()}
    
    tp_examples = []
    fp_examples = []
    fn_examples = []
    
    val_pred_sets = {k: set(v) for k, v in val_preds.items()}
    
    for s1_id, true_set in val_gt.items():
        pred_set = val_pred_sets.get(s1_id, set())
        s1_info = s1_dict.get(s1_id, {})
        
        # True Positives
        for cid in true_set.intersection(pred_set):
            if len(tp_examples) < 5:
                c_info = s23_dict.get(cid, {})
                tp_examples.append({
                    "s1_id": s1_id, "cand_id": cid,
                    "s1_name": s1_info.get("business_name"), "cand_name": c_info.get("business_name"),
                    "s1_addr": s1_info.get("business_address"), "cand_addr": c_info.get("business_address"),
                    "type": "TRUE_POSITIVE"
                })
                
        # False Positives (False Merges)
        for cid in pred_set - true_set:
            if len(fp_examples) < 5:
                c_info = s23_dict.get(cid, {})
                fp_examples.append({
                    "s1_id": s1_id, "cand_id": cid,
                    "s1_name": s1_info.get("business_name"), "cand_name": c_info.get("business_name"),
                    "s1_addr": s1_info.get("business_address"), "cand_addr": c_info.get("business_address"),
                    "type": "FALSE_POSITIVE"
                })
                
        # False Negatives (Missed matches)
        for cid in true_set - pred_set:
            if len(fn_examples) < 5:
                c_info = s23_dict.get(cid, {})
                fn_examples.append({
                    "s1_id": s1_id, "cand_id": cid,
                    "s1_name": s1_info.get("business_name"), "cand_name": c_info.get("business_name"),
                    "s1_addr": s1_info.get("business_address"), "cand_addr": c_info.get("business_address"),
                    "type": "FALSE_NEGATIVE"
                })
                
    error_analysis_data = {
        "true_positives_sample": tp_examples,
        "false_positives_sample": fp_examples,
        "false_negatives_sample": fn_examples
    }
    
    save_json(error_analysis_data, os.path.join(reports_dir, "error_analysis.json"))
    logger.info(f"Saved error analysis report to {os.path.join(reports_dir, 'error_analysis.json')}")
    return error_analysis_data
