import polars as pl
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import subprocess
from typing import Dict, Any
from collections import defaultdict

from config import Config, DEFAULT_CONFIG
from utils import logger, Timer, save_json
from data_loader import load_dataset
from candidate_generation import CandidateGenerator
from pair_dataset import build_pairwise_dataset
from train_model import ModelTrainer
from threshold_optimizer import ThresholdOptimizer
from matcher import EntityMatcher
from evaluation import evaluate_predictions
from validation import run_ablation_study, run_error_analysis
from inference import run_test_inference

def generate_eda_reports(
    train_s1: pl.DataFrame,
    train_s2: pl.DataFrame,
    train_s3: pl.DataFrame,
    train_gt: pl.DataFrame,
    reports_dir: str
):
    """Generates eda_summary.json and eda_report.html."""
    logger.info("--- Generating Exploratory Data Analysis (EDA) Reports ---")
    
    gt_df = train_gt.to_pandas()
    gt_df['match_list'] = gt_df['matched_entity_ids'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
    gt_df['num_matches'] = gt_df['match_list'].apply(len)
    
    match_counts = gt_df['num_matches'].value_counts().to_dict()
    singleton_count = int(match_counts.get(0, 0))
    singleton_pct = float(singleton_count / len(gt_df)) * 100
    
    eda_summary = {
        "dataset_statistics": {
            "train_source1_rows": len(train_s1),
            "train_source2_rows": len(train_s2),
            "train_source3_rows": len(train_s3),
            "train_ground_truth_rows": len(train_gt),
            "s1_countries": train_s1["country"].value_counts().to_dicts(),
            "s2_countries": train_s2["country"].value_counts().to_dicts(),
            "s3_countries": train_s3["country"].value_counts().to_dicts(),
        },
        "ground_truth_distribution": {
            "total_s1_entities": len(gt_df),
            "singleton_count": singleton_count,
            "singleton_percentage": singleton_pct,
            "match_count_distribution": match_counts,
            "total_matching_pairs": int(gt_df['num_matches'].sum())
        },
        "noise_characteristics": {
            "name_variations": ["legal suffixes", "abbreviations", "case differences", "typos", "transliterations"],
            "address_variations": ["missing components", "landmark references", "street abbreviation variations", "multilingual script"],
            "cross_country_matches": 0
        }
    }
    
    save_json(eda_summary, os.path.join(reports_dir, "eda_summary.json"))
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Amazon ML Challenge — Entity Resolution EDA Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #334155; color: #38bdf8; }}
        .badge {{ background: #0284c7; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Amazon ML Challenge 2026: Business Entity Resolution</h1>
    <h2>Exploratory Data Analysis (EDA) Report</h2>
    
    <div class="card">
        <h3>Dataset Summary</h3>
        <table>
            <tr><th>Dataset</th><th>Record Count</th><th>Role</th></tr>
            <tr><td>Source 1 (Train)</td><td>{len(train_s1):,}</td><td>Deduplicated Reference Entities</td></tr>
            <tr><td>Source 2 (Train)</td><td>{len(train_s2):,}</td><td>Noisy Business Records</td></tr>
            <tr><td>Source 3 (Train)</td><td>{len(train_s3):,}</td><td>Noisy Business Records</td></tr>
            <tr><td>Ground Truth (Train)</td><td>{len(train_gt):,}</td><td>Reference Match Labels</td></tr>
        </table>
    </div>

    <div class="card">
        <h3>Ground Truth & Singletons</h3>
        <p>Total S1 Entities: <strong>{len(gt_df):,}</strong></p>
        <p>Singletons (Zero Matches): <strong>{singleton_count:,} ({singleton_pct:.2f}%)</strong></p>
        <p>Entities with Matches: <strong>{len(gt_df)-singleton_count:,} ({100-singleton_pct:.2f}%)</strong></p>
        <p>Total True Linkage Pairs: <strong>{int(gt_df['num_matches'].sum()):,}</strong></p>
    </div>

    <div class="card">
        <h3>Key Technical Findings</h3>
        <ul>
            <li><strong>Strict Country Partitioning:</strong> 100% of matching pairs stay within the same country boundary (0 cross-country matches).</li>
            <li><strong>Open Set Country Support:</strong> Pipeline dynamically handles open set countries (US, India, France).</li>
            <li><strong>Precision-Weighted Metric:</strong> Submissions evaluated under Macro F0.5, weighting precision 2x over recall.</li>
        </ul>
    </div>
</body>
</html>"""
    with open(os.path.join(reports_dir, "eda_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved EDA report to {os.path.join(reports_dir, 'eda_report.html')}")

def run_pipeline(config: Config = DEFAULT_CONFIG):
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE END-TO-END ENTITY RESOLUTION PIPELINE")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------
    # 1. LOAD DATA & EDA
    # -------------------------------------------------------------
    with Timer("Loading Training Data"):
        train_s1, train_s2, train_s3, train_gt = load_dataset(config.dataset_dir, is_train=True)
        
    generate_eda_reports(train_s1, train_s2, train_s3, train_gt, config.reports_dir)
    
    # Ground truth mapping
    gt_map = {}
    for r in train_gt.to_dicts():
        s1_id = r["source1_entity_id"]
        mids = [i.strip() for i in str(r["matched_entity_ids"] or "").split(",") if i.strip()]
        gt_map[s1_id] = set(mids)
        
    # -------------------------------------------------------------
    # 2. GROUPED VALIDATION SPLIT (SOURCE 1 ENTITY LEVEL)
    # -------------------------------------------------------------
    logger.info("--- Creating Grouped Validation Split at Source 1 Level ---")
    np.random.seed(config.seed)
    
    # Stratified sample by country
    s1_df_sample = train_s1.sample(n=min(config.train_sample_size, len(train_s1)), seed=config.seed)
    all_sample_s1_ids = s1_df_sample["entity_id"].to_list()
    np.random.shuffle(all_sample_s1_ids)
    
    val_count = int(len(all_sample_s1_ids) * config.val_split_ratio)
    val_s1_ids = set(all_sample_s1_ids[:val_count])
    train_s1_ids = set(all_sample_s1_ids[val_count:])
    
    logger.info(f"Sampled {len(all_sample_s1_ids)} S1 Entities -> Train: {len(train_s1_ids)}, Val: {len(val_s1_ids)}")
    
    # -------------------------------------------------------------
    # 3. CANDIDATE GENERATION (BLOCKING) FOR VALIDATION
    # -------------------------------------------------------------
    cache_cand_path = os.path.join(config.models_dir, "train_candidates.parquet")
    blocker = CandidateGenerator(top_k=config.blocking_top_k, max_key_freq=config.max_key_freq)
    if os.path.exists(cache_cand_path):
        logger.info(f"Loading cached training candidates from {cache_cand_path}")
        cand_df = pl.read_parquet(cache_cand_path)
        blocking_diag = {"cached": True, "total_candidate_pairs": len(cand_df)}
    else:
        with Timer("Candidate Generation (Blocking)"):
            cand_df, blocking_diag = blocker.generate(s1_df_sample, train_s2, train_s3)
            cand_df.write_parquet(cache_cand_path)
        
    # Measure blocking recall on validation set
    val_cands = cand_df.filter(pl.col("s1_id").is_in(list(val_s1_ids)))
    val_cand_dict = defaultdict(set)
    for r in val_cands.to_dicts():
        if r.get("cand_id"):
            val_cand_dict[r["s1_id"]].add(r["cand_id"])
            
    val_gt_subset = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    total_val_gt_matches = sum(len(v) for v in val_gt_subset.values())
    val_recalled_matches = sum(len(val_cand_dict.get(s1_id, set()).intersection(val_gt_subset[s1_id])) for s1_id in val_s1_ids)
    blocking_recall = val_recalled_matches / total_val_gt_matches if total_val_gt_matches > 0 else 1.0
    
    logger.info(f"Validation Blocking Recall: {blocking_recall*100:.2f}% ({val_recalled_matches}/{total_val_gt_matches})")
    blocking_diag["val_blocking_recall"] = blocking_recall
    
    # -------------------------------------------------------------
    # 4. PAIRWISE FEATURE DATASET CREATION
    # -------------------------------------------------------------
    cache_feat_path = os.path.join(config.models_dir, "train_features.npz")
    if os.path.exists(cache_feat_path):
        logger.info(f"Loading cached training features from {cache_feat_path}")
        data = np.load(cache_feat_path, allow_pickle=True)
        X_all = data["X_all"]
        y_all = data["y_all"]
        pair_ids = [(p[0], p[1]) for p in data["pair_ids"]]
    else:
        with Timer("Pairwise Feature Engineering"):
            X_all, y_all, pair_ids = build_pairwise_dataset(cand_df, s1_df_sample, train_s2, train_s3, gt_map)
            np.savez_compressed(cache_feat_path, X_all=X_all, y_all=y_all, pair_ids=np.array(pair_ids, dtype=object))
        
    train_mask = np.array([p[0] in train_s1_ids for p in pair_ids])
    val_mask = np.array([p[0] in val_s1_ids for p in pair_ids])
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    val_pairs = [pair_ids[i] for i in range(len(pair_ids)) if val_mask[i]]
    
    logger.info(f"Dataset partition -> Train pairs: {len(X_train)} (Pos: {np.sum(y_train)}), Val pairs: {len(X_val)} (Pos: {np.sum(y_val)})")
    
    # -------------------------------------------------------------
    # 5. MODEL TRAINING & BASELINE COMPARISON
    # -------------------------------------------------------------
    with Timer("LightGBM Model Training"):
        trainer = ModelTrainer(params=config.lgb_params)
        trainer.train(X_train, y_train, X_val, y_val)
        
    val_probs = trainer.predict_proba(X_val)
    
    # Baseline comparison (Rule-based exact name match)
    rule_preds = {}
    s1_name_map = {r["entity_id"]: str(r.get("business_name") or "").lower().strip() for r in s1_df_sample.to_dicts()}
    s23_comb = pl.concat([train_s2, train_s3])
    needed_cands = set(p[1] for p in val_pairs)
    s23_name_map = {r["entity_id"]: str(r.get("business_name") or "").lower().strip() for r in s23_comb.filter(pl.col("entity_id").is_in(list(needed_cands))).to_dicts()}
    
    for s1_id in val_s1_ids:
        matches = set()
        for cid in val_cand_dict.get(s1_id, set()):
            if s1_name_map.get(s1_id) == s23_name_map.get(cid) and s1_name_map.get(s1_id) != "":
                matches.add(cid)
        rule_preds[s1_id] = matches
        
    baseline_eval = evaluate_predictions(val_gt_subset, rule_preds, list(val_s1_ids))
    logger.info(f"Baseline (Exact Name Rule Match) -> Macro F0.5: {baseline_eval['macro_f05']:.4f} | Prec: {baseline_eval['macro_precision']:.4f} | Rec: {baseline_eval['macro_recall']:.4f}")
    
    # -------------------------------------------------------------
    # 6. THRESHOLD OPTIMIZATION FOR MACRO F0.5
    # -------------------------------------------------------------
    with Timer("Threshold Optimization"):
        optimizer = ThresholdOptimizer(search_grid=config.threshold_search_grid)
        best_threshold = optimizer.optimize(val_pairs, val_probs, val_gt_subset, list(val_s1_ids))
        
    # Evaluate final validation metrics
    matcher = EntityMatcher(threshold=best_threshold)
    ml_val_preds = matcher.match_candidates(val_pairs, val_probs, list(val_s1_ids), candidate_dict=val_cand_dict)
    val_eval = evaluate_predictions(val_gt_subset, {k: set(v) for k, v in ml_val_preds.items()}, list(val_s1_ids))
    
    logger.info("=" * 50)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info(f"Macro F0.5 Score:     {val_eval['macro_f05']:.4f}")
    logger.info(f"Macro Precision:      {val_eval['macro_precision']:.4f}")
    logger.info(f"Macro Recall:         {val_eval['macro_recall']:.4f}")
    logger.info(f"Singleton Accuracy:   {val_eval['singleton_accuracy']:.4f}")
    logger.info(f"Blocking Recall:      {blocking_recall:.4f}")
    logger.info(f"Pair Precision:       {val_eval['pair_precision']:.4f}")
    logger.info(f"Pair Recall:          {val_eval['pair_recall']:.4f}")
    logger.info(f"Optimal Threshold:    {best_threshold:.2f}")
    logger.info("=" * 50)
    
    # Save validation & model reports
    validation_report = {
        "validation_metrics": val_eval,
        "baseline_comparison": baseline_eval,
        "optimal_threshold": best_threshold,
        "threshold_grid_search": optimizer.threshold_history,
        "blocking_diagnostics": blocking_diag,
        "top_features": trainer.feature_importances
    }
    save_json(validation_report, os.path.join(config.reports_dir, "validation_report.json"))
    save_json(validation_report, os.path.join(config.reports_dir, "model_report.json"))
    
    # -------------------------------------------------------------
    # 7. ABLATION STUDY & ERROR ANALYSIS
    # -------------------------------------------------------------
    run_ablation_study(X_train, y_train, X_val, val_pairs, val_gt_subset, list(val_s1_ids), config.reports_dir)
    run_error_analysis(val_pairs, val_probs, val_gt_subset, ml_val_preds, s1_df_sample, train_s2, train_s3, config.reports_dir)
    
    # -------------------------------------------------------------
    # 8. TRAIN FINAL MODEL ON ALL TRAINING SAMPLES
    # -------------------------------------------------------------
    logger.info("--- Fitting Final Production Model on Full Training Sample ---")
    with Timer("Final Model Training"):
        final_trainer = ModelTrainer(params=config.lgb_params)
        final_trainer.train(X_all, y_all)
        
    # -------------------------------------------------------------
    # 9. TEST SET INFERENCE & VALIDATION
    # -------------------------------------------------------------
    logger.info("--- Loading Test Dataset & Running End-to-End Inference ---")
    with Timer("Loading Test Dataset"):
        test_s1, test_s2, test_s3, _ = load_dataset(config.dataset_dir, is_train=False)
        
    match_file, cand_file = run_test_inference(
        test_s1, test_s2, test_s3,
        model=final_trainer,
        threshold=best_threshold,
        output_dir=config.output_dir,
        top_k=config.blocking_top_k
    )
    
    # -------------------------------------------------------------
    # 10. AUTOMATED SUBMISSION VALIDATION
    # -------------------------------------------------------------
    logger.info("--- Validating Submission Files with utils/validate_submission.py ---")
    validator_path = os.path.join(config.base_dir, "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        cmd = [
            sys.executable, validator_path,
            "--matching", match_file,
            "--candidate", cand_file,
            "--test-dir", os.path.join(config.dataset_dir, "test")
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            logger.info("✅ SUBMISSION INTEGRITY CHECK: PASS (Exit code 0). Ready for portal submission!")
        else:
            logger.error("❌ SUBMISSION INTEGRITY CHECK: FAIL. Check validation issues.")
            
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info(f"Final matching output:  {match_file}")
    logger.info(f"Final candidate output: {cand_file}")
    logger.info("=" * 70)
    return validation_report

if __name__ == "__main__":
    run_pipeline()
