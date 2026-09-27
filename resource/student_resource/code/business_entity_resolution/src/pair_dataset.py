import numpy as np
import polars as pl
from typing import Dict, Set, Tuple, List
from feature_engineering import extract_pairwise_features, FEATURE_NAMES

def build_pairwise_dataset(
    candidate_df: pl.DataFrame,
    df_s1: pl.DataFrame,
    df_s2: pl.DataFrame,
    df_s3: pl.DataFrame,
    gt_map: Dict[str, Set[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds feature matrix X and label array y for all candidate pairs in candidate_df.
    Returns: (X, y, pair_ids) where pair_ids is list of (s1_id, cand_id).
    """
    valid_cands = candidate_df.filter(pl.col("cand_id").is_not_null())
    cand_rows = valid_cands.to_dicts()
    
    # Entity dict lookups
    s1_dict = {r["entity_id"]: r for r in df_s1.to_dicts()}
    
    needed_cand_ids = set(valid_cands["cand_id"])
    s23_combined = pl.concat([df_s2, df_s3])
    s23_sub = s23_combined.filter(pl.col("entity_id").is_in(list(needed_cand_ids)))
    s23_dict = {r["entity_id"]: r for r in s23_sub.to_dicts()}
    
    X = []
    y = []
    pair_ids = []
    
    for r in cand_rows:
        s1_id = r["s1_id"]
        cand_id = r["cand_id"]
        score = r["blocking_score"] if "blocking_score" in r else r.get("score", 0.0)
        
        s1_info = s1_dict.get(s1_id)
        cand_info = s23_dict.get(cand_id)
        if not s1_info or not cand_info:
            continue
            
        feat = extract_pairwise_features(
            s1_info.get("business_name"),
            s1_info.get("business_address"),
            s1_info.get("country"),
            cand_id,
            cand_info.get("business_name"),
            cand_info.get("business_address"),
            cand_info.get("country"),
            score
        )
        
        X.append(feat)
        pair_ids.append((s1_id, cand_id))
        
        if gt_map is not None:
            label = 1 if cand_id in gt_map.get(s1_id, set()) else 0
            y.append(label)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32) if gt_map is not None else None, pair_ids
