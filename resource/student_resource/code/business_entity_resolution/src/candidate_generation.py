import polars as pl
import numpy as np
import time
from typing import Tuple, Dict, Any
from utils import logger, Timer
from preprocessing import preprocess_dataframe

class CandidateGenerator:
    """
    Production-grade Multi-Strategy Candidate Generation (Blocking) Engine.
    
    Strategies implemented:
    - Strategy A: Normalized Name word tokens (nw)
    - Strategy B: Address locality / building word tokens (aw)
    - Strategy C: Exact house / street number & PIN/zip anchors (d)
    - Strategy D: Country-partitioned matching
    """
    def __init__(self, top_k: int = 50, max_key_freq: int = 2000):
        self.top_k = top_k
        self.max_key_freq = max_key_freq
        self.diagnostics = {}

    def extract_blocking_keys(self, df: pl.DataFrame, id_col: str) -> pl.DataFrame:
        prep = preprocess_dataframe(df)
        
        # 1. Name Word Tokens (len >= 3)
        nw = prep.select([
            pl.col(id_col),
            pl.col("business_name_normalized").str.split(" ").alias("w")
        ]).explode("w").filter(pl.col("w").str.len_chars() >= 3).with_columns(
            ("nw:" + pl.col("w")).alias("keys")
        ).select([id_col, "keys"])
        
        # 2. Address Word Tokens (len >= 4)
        aw = prep.select([
            pl.col(id_col),
            pl.col("business_address_normalized").str.split(" ").alias("w")
        ]).explode("w").filter(pl.col("w").str.len_chars() >= 4).with_columns(
            ("aw:" + pl.col("w")).alias("keys")
        ).select([id_col, "keys"])
        
        # 3. Numeric Digits (house #, street #, PIN codes: 3 to 6 digits)
        dig = prep.select([
            pl.col(id_col),
            (pl.col("business_name_raw") + " " + pl.col("business_address_raw")).str.split(" ").alias("w")
        ]).explode("w").filter(
            pl.col("w").cast(pl.Int64, strict=False).is_not_null() & 
            (pl.col("w").str.len_chars() >= 3) & 
            (pl.col("w").str.len_chars() <= 6)
        ).with_columns(
            ("d:" + pl.col("w")).alias("keys")
        ).select([id_col, "keys"])
        
        combined = pl.concat([nw, aw, dig])
        return combined

    def generate(
        self,
        df_s1: pl.DataFrame,
        df_s2: pl.DataFrame,
        df_s3: pl.DataFrame
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Executes country-partitioned multi-channel candidate generation.
        Returns candidate pairs dataframe and blocking diagnostics dict.
        """
        start_time = time.time()
        countries = df_s1["country"].fill_null("UNKNOWN").unique().to_list()
        
        total_possible_pairs = len(df_s1) * (len(df_s2) + len(df_s3))
        all_country_cands = []
        
        logger.info(f"Generating candidate pairs for {len(df_s1)} S1 entities across {len(countries)} countries: {countries}")
        
        for c in countries:
            c_start = time.time()
            s1_c = df_s1.filter(pl.col("country").fill_null("UNKNOWN") == c)
            s2_c = df_s2.filter(pl.col("country").fill_null("UNKNOWN") == c)
            s3_c = df_s3.filter(pl.col("country").fill_null("UNKNOWN") == c)
            s23_c = pl.concat([s2_c, s3_c])
            
            if len(s1_c) == 0 or len(s23_c) == 0:
                continue
                
            # Candidate inverted index & IDF weights
            cand_keys = self.extract_blocking_keys(s23_c, "entity_id").rename({"entity_id": "cand_id"})
            key_counts = cand_keys.group_by("keys").len().filter(pl.col("len") <= self.max_key_freq)
            key_weights = key_counts.with_columns(
                (1.0 / (1.0 + pl.col("len").log())).alias("weight")
            ).select(["keys", "weight"])
            
            cand_keys_weighted = cand_keys.join(key_weights, on="keys", how="inner")
            
            # Query S1 keys & join with candidate index (chunked if large to bound memory)
            if len(s1_c) > 50000:
                chunk_size = 50000
                num_chunks = int(np.ceil(len(s1_c) / chunk_size))
                s1_c_list = []
                for ch_i in range(num_chunks):
                    ch_s1 = s1_c.slice(ch_i * chunk_size, chunk_size)
                    ch_keys = self.extract_blocking_keys(ch_s1, "entity_id").rename({"entity_id": "s1_id"})
                    ch_joined = ch_keys.join(cand_keys_weighted, on="keys", how="inner")
                    ch_scores = ch_joined.group_by(["s1_id", "cand_id"]).agg(pl.col("weight").sum().alias("blocking_score"))
                    ch_top = (
                        ch_scores
                        .sort(["s1_id", "blocking_score"], descending=[False, True])
                        .filter(pl.int_range(0, pl.len()).over("s1_id") < self.top_k)
                    )
                    all_ch_ids = ch_s1.select(pl.col("entity_id").alias("s1_id")).unique()
                    ch_top = all_ch_ids.join(ch_top, on="s1_id", how="left")
                    s1_c_list.append(ch_top)
                top_cands = pl.concat(s1_c_list)
            else:
                s1_keys = self.extract_blocking_keys(s1_c, "entity_id").rename({"entity_id": "s1_id"})
                joined = s1_keys.join(cand_keys_weighted, on="keys", how="inner")
                scores = joined.group_by(["s1_id", "cand_id"]).agg(pl.col("weight").sum().alias("blocking_score"))
                top_cands = (
                    scores
                    .sort(["s1_id", "blocking_score"], descending=[False, True])
                    .filter(pl.int_range(0, pl.len()).over("s1_id") < self.top_k)
                )
                all_s1_ids = s1_c.select(pl.col("entity_id").alias("s1_id")).unique()
                top_cands = all_s1_ids.join(top_cands, on="s1_id", how="left")
            
            all_country_cands.append(top_cands)
            logger.info(f"Country '{c}' candidate generation: {len(s1_c)} S1 entities against {len(s23_c)} candidates in {time.time()-c_start:.2f}s")
            
        combined_cands = pl.concat(all_country_cands)
        total_cand_pairs = len(combined_cands.filter(pl.col("cand_id").is_not_null()))
        reduction_ratio = 1.0 - (total_cand_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0
        avg_cands_per_s1 = total_cand_pairs / len(df_s1) if len(df_s1) > 0 else 0.0
        
        self.diagnostics = {
            "total_s1_entities": len(df_s1),
            "total_s2_records": len(df_s2),
            "total_s3_records": len(df_s3),
            "total_possible_pairs": total_possible_pairs,
            "total_candidate_pairs": total_cand_pairs,
            "candidate_reduction_ratio": reduction_ratio,
            "avg_candidates_per_s1": avg_cands_per_s1,
            "blocking_time_seconds": time.time() - start_time
        }
        
        logger.info(f"Blocking complete: {total_cand_pairs} candidate pairs | Reduction Ratio: {reduction_ratio*100:.6f}% | Avg {avg_cands_per_s1:.1f} cands/S1 in {self.diagnostics['blocking_time_seconds']:.2f}s")
        return combined_cands, self.diagnostics
