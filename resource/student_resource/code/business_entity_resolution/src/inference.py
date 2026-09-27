import polars as pl
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, Set, List, Tuple
from collections import defaultdict

from utils import logger, Timer
from candidate_generation import CandidateGenerator
from feature_engineering import extract_pairwise_features
from train_model import ModelTrainer
from matcher import EntityMatcher

def run_test_inference(
    df_test_s1: pl.DataFrame,
    df_test_s2: pl.DataFrame,
    df_test_s3: pl.DataFrame,
    model: ModelTrainer,
    threshold: float,
    output_dir: str,
    top_k: int = 50
) -> Tuple[str, str]:
    """
    Executes end-to-end inference on the full test set.
    Produces:
    1. output/candidate_pairs.tsv
    2. output/matching_results.tsv
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Candidate Generation (Blocking)
    # ---------------------------------------------------------
    logger.info("--- Generating Candidates for Full Test Set ---")
    with Timer("Test Set Blocking"):
        blocker = CandidateGenerator(top_k=top_k)
        test_cands, _ = blocker.generate(df_test_s1, df_test_s2, df_test_s3)
        
    # Group candidate IDs per S1 entity
    valid_test_cands = test_cands.filter(pl.col("cand_id").is_not_null())
    grouped_cands = (
        valid_test_cands
        .group_by("s1_id")
        .agg(pl.col("cand_id").unique().str.concat(",").alias("candidate_entity_ids"))
        .rename({"s1_id": "source1_entity_id"})
    )
    
    # Ensure every single S1 entity in test_source1.tsv appears on exactly one row
    all_s1_df = df_test_s1.select(pl.col("entity_id").alias("source1_entity_id")).unique()
    final_cand_df = all_s1_df.join(grouped_cands, on="source1_entity_id", how="left").with_columns(
        pl.col("candidate_entity_ids").fill_null("")
    ).sort("source1_entity_id")
    
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    final_cand_df.to_pandas().to_csv(cand_path, sep="\t", index=False)
    logger.info(f"Saved {len(final_cand_df)} rows to candidate file: {cand_path}")
    
    # Build candidate lookup set for post-processing consistency check
    cand_map = {}
    for r in final_cand_df.to_dicts():
        c_list = [i.strip() for i in str(r["candidate_entity_ids"] or "").split(",") if i.strip()]
        cand_map[r["source1_entity_id"]] = set(c_list)
        
    # ---------------------------------------------------------
    # 2. Batch Feature Extraction & Scoring
    # ---------------------------------------------------------
    logger.info("--- Extracting Features & Scoring Test Candidate Pairs ---")
    s1_join_df = df_test_s1.select([
        pl.col("entity_id").alias("s1_id"),
        pl.col("business_name").alias("s1_name"),
        pl.col("business_address").alias("s1_addr"),
        pl.col("country").alias("s1_country")
    ])
    s23_combined = pl.concat([
        df_test_s2.select(["entity_id", "business_name", "business_address", "country"]),
        df_test_s3.select(["entity_id", "business_name", "business_address", "country"])
    ]).unique(subset=["entity_id"])
    s23_join_df = s23_combined.select([
        pl.col("entity_id").alias("cand_id"),
        pl.col("business_name").alias("cand_name"),
        pl.col("business_address").alias("cand_addr"),
        pl.col("country").alias("cand_country")
    ])
    
    total_pairs = len(valid_test_cands)
    logger.info(f"Total candidate pairs to score: {total_pairs}")
    
    batch_size = 50000
    all_pair_ids = []
    all_probs = []
    
    with Timer("Test Set Scoring"):
        for i in range(0, total_pairs, batch_size):
            cand_slice = valid_test_cands.slice(i, batch_size)
            batch = cand_slice.join(s1_join_df, on="s1_id", how="left").join(s23_join_df, on="cand_id", how="left")
            
            s1_ids = batch["s1_id"].to_list()
            c_ids = batch["cand_id"].to_list()
            s1_names = batch["s1_name"].to_list()
            s1_addrs = batch["s1_addr"].to_list()
            s1_countries = batch["s1_country"].to_list()
            c_names = batch["cand_name"].to_list()
            c_addrs = batch["cand_addr"].to_list()
            c_countries = batch["cand_country"].to_list()
            scores = batch["blocking_score"].fill_null(0.0).to_list() if "blocking_score" in batch.columns else [0.0] * len(batch)
            
            X_batch = [
                extract_pairwise_features(n1, a1, c1, cid, n2, a2, c2, sc)
                for n1, a1, c1, cid, n2, a2, c2, sc in zip(
                    s1_names, s1_addrs, s1_countries, c_ids,
                    c_names, c_addrs, c_countries, scores
                )
            ]
            
            if X_batch:
                probs = model.predict_proba(np.array(X_batch, dtype=np.float32))
                batch_pairs = list(zip(s1_ids, c_ids))
                all_pair_ids.extend(batch_pairs)
                all_probs.extend(probs)
                
    # ---------------------------------------------------------
    # 3. Match Decision & Formatting
    # ---------------------------------------------------------
    logger.info("--- Formatting matching_results.tsv ---")
    all_test_s1_ids = sorted(df_test_s1["entity_id"].to_list())
    matcher = EntityMatcher(threshold=threshold)
    match_predictions = matcher.match_candidates(
        all_pair_ids,
        np.array(all_probs),
        all_test_s1_ids,
        candidate_dict=cand_map
    )
    
    match_rows = []
    for s1_id in all_test_s1_ids:
        mids = match_predictions.get(s1_id, [])
        match_str = ",".join(mids)
        match_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": match_str})
        
    final_match_df = pd.DataFrame(match_rows)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    final_match_df.to_csv(match_path, sep="\t", index=False)
    logger.info(f"Saved {len(final_match_df)} rows to matching file: {match_path}")
    
    return match_path, cand_path
