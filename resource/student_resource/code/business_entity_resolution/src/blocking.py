import polars as pl
import numpy as np
import time
from preprocessor import STOPWORDS_REGEX, ADDR_FILLERS_REGEX

class MultiChannelBlocking:
    """Fast Multi-Channel Inverted Index Candidate Generation in Polars.
    
    Generates blocking candidates partitioned by country.
    Supports open-set country values (US, India, France, etc.).
    """
    def __init__(self, top_k: int = 50, max_key_freq: int = 15000):
        self.top_k = top_k
        self.max_key_freq = max_key_freq

    def extract_keys(self, df: pl.DataFrame, id_col: str) -> pl.DataFrame:
        clean_df = df.with_columns([
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(STOPWORDS_REGEX, "").alias("name_clean"),
            pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(ADDR_FILLERS_REGEX, "").alias("addr_clean"),
        ])
        
        # Channel 1: Name Word Tokens
        nw = clean_df.select([
            pl.col(id_col),
            pl.col("name_clean").str.extract_all(r'\b[a-z0-9]{3,}\b').alias("k")
        ]).explode("k").filter(pl.col("k").is_not_null()).with_columns(("nw:" + pl.col("k")).alias("keys")).select([id_col, "keys"])
        
        # Channel 2: Address Word Tokens
        aw = clean_df.select([
            pl.col(id_col),
            pl.col("addr_clean").str.extract_all(r'\b[a-z]{3,}\b').alias("k")
        ]).explode("k").filter(pl.col("k").is_not_null()).with_columns(("aw:" + pl.col("k")).alias("keys")).select([id_col, "keys"])
        
        # Channel 3: Digits from Name + Address
        dig = clean_df.select([
            pl.col(id_col),
            (pl.col("name_clean") + " " + pl.col("addr_clean")).str.extract_all(r'\b\d{1,6}\b').alias("k")
        ]).explode("k").filter(pl.col("k").is_not_null()).with_columns(("d:" + pl.col("k")).alias("keys")).select([id_col, "keys"])
        
        # Channel 4: Name 3-gram Prefixes
        n3p = clean_df.select([
            pl.col(id_col),
            pl.col("name_clean").str.extract_all(r'\b[a-z0-9]{3,}\b').alias("words")
        ]).explode("words").filter(pl.col("words").is_not_null()).with_columns(
            ("n3p:" + pl.col("words").str.slice(0, 3)).alias("keys")
        ).select([id_col, "keys"])

        return pl.concat([nw, aw, dig, n3p]).unique()

    def generate_candidates(self, df_s1: pl.DataFrame, df_s2: pl.DataFrame, df_s3: pl.DataFrame) -> pl.DataFrame:
        """Run candidate generation per country and return candidate pairs dataframe."""
        countries = df_s1["country"].unique().to_list()
        all_country_results = []
        
        for c in countries:
            t0 = time.time()
            s1_c = df_s1.filter(pl.col("country") == c)
            s2_c = df_s2.filter(pl.col("country") == c)
            s3_c = df_s3.filter(pl.col("country") == c)
            
            s23_c = pl.concat([s2_c, s3_c])
            if len(s1_c) == 0 or len(s23_c) == 0:
                continue
                
            # Extract candidate keys & compute IDF weights
            cand_keys = self.extract_keys(s23_c, "entity_id").rename({"entity_id": "cand_id"})
            key_counts = cand_keys.group_by("keys").len().filter(pl.col("len") <= self.max_key_freq)
            key_weights = key_counts.with_columns(
                (1.0 / (1.0 + pl.col("len").log())).alias("weight")
            ).select(["keys", "weight"])
            
            cand_keys_weighted = cand_keys.join(key_weights, on="keys", how="inner")
            
            # Extract S1 keys & join
            s1_keys = self.extract_keys(s1_c, "entity_id").rename({"entity_id": "s1_id"})
            joined = s1_keys.join(cand_keys_weighted, on="keys", how="inner")
            
            scores = joined.group_by(["s1_id", "cand_id"]).agg(pl.col("weight").sum().alias("score"))
            
            # Select top K candidates per S1 entity
            top_cands = (
                scores
                .sort(["s1_id", "score"], descending=[False, True])
                .group_by("s1_id")
                .head(self.top_k)
            )
            
            # Ensure every S1 entity in this country has a row even if 0 candidates found
            all_s1_ids = s1_c.select(pl.col("entity_id").alias("s1_id")).unique()
            top_cands = all_s1_ids.join(top_cands, on="s1_id", how="left")
            
            all_country_results.append(top_cands)
            print(f"Country '{c}': Generated candidates for {len(s1_c)} S1 entities against {len(s23_c)} S2/S3 records in {time.time()-t0:.2f}s", flush=True)
            
        combined_cands = pl.concat(all_country_results)
        return combined_cands
