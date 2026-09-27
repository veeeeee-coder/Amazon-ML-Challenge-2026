import os
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

FEATURE_NAMES = [
    "name_exact_match", "name_norm_equality", "name_fuzz_ratio", "name_partial_ratio",
    "name_token_sort", "name_token_set", "name_wratio", "name_jaccard",
    "name_len_diff", "name_token_count_diff", "name_prefix_sim", "name_suffix_sim",
    "addr_exact_match", "addr_fuzz_ratio", "addr_partial_ratio", "addr_token_sort",
    "addr_token_set", "addr_jaccard", "addr_len_diff", "addr_missing_flag",
    "dig_jaccard", "dig_exact_match", "dig_count_shared",
    "comb_token_set", "country_exact_match", "is_s2", "blocking_score"
]

def fast_pairwise_features(s1_n_norm, s1_a_norm, s1_country, cid, c_n_norm, c_a_norm, c_country, blocking_score=1.0):
    n_exact = 1.0 if s1_n_norm == c_n_norm and s1_n_norm != "" else 0.0
    n_fuzz = fuzz.ratio(s1_n_norm, c_n_norm)
    n_part = fuzz.partial_ratio(s1_n_norm, c_n_norm)
    n_tsort = fuzz.token_sort_ratio(s1_n_norm, c_n_norm)
    n_tset = fuzz.token_set_ratio(s1_n_norm, c_n_norm)
    n_wratio = fuzz.WRatio(s1_n_norm, c_n_norm)
    
    s1_n_toks = set(s1_n_norm.split())
    c_n_toks = set(c_n_norm.split())
    n_jaccard = len(s1_n_toks.intersection(c_n_toks)) / len(s1_n_toks.union(c_n_toks)) if s1_n_toks and c_n_toks else 0.0
    n_len_diff = abs(len(s1_n_norm) - len(c_n_norm))
    n_tok_diff = abs(len(s1_n_toks) - len(c_n_toks))
    n_pre_sim = 1.0 if s1_n_norm[:4] == c_n_norm[:4] and s1_n_norm != "" else 0.0
    n_suf_sim = 1.0 if s1_n_norm[-4:] == c_n_norm[-4:] and s1_n_norm != "" else 0.0
    
    addr_missing = 1.0 if (not s1_a_norm or not c_a_norm) else 0.0
    if addr_missing == 0.0:
        a_exact = 1.0 if s1_a_norm == c_a_norm else 0.0
        a_fuzz = fuzz.ratio(s1_a_norm, c_a_norm)
        a_part = fuzz.partial_ratio(s1_a_norm, c_a_norm)
        a_tsort = fuzz.token_sort_ratio(s1_a_norm, c_a_norm)
        a_tset = fuzz.token_set_ratio(s1_a_norm, c_a_norm)
        s1_a_toks = set(s1_a_norm.split())
        c_a_toks = set(c_a_norm.split())
        a_jaccard = len(s1_a_toks.intersection(c_a_toks)) / len(s1_a_toks.union(c_a_toks)) if s1_a_toks and c_a_toks else 0.0
        a_len_diff = abs(len(s1_a_norm) - len(c_a_norm))
    else:
        a_exact = a_fuzz = a_part = a_tsort = a_tset = a_jaccard = 0.0
        a_len_diff = abs(len(s1_a_norm) - len(c_a_norm))
        
    s1_digs = set([w for w in s1_a_norm.split() if w.isdigit()])
    c_digs = set([w for w in c_a_norm.split() if w.isdigit()])
    if s1_digs and c_digs:
        dig_shared = len(s1_digs.intersection(c_digs))
        dig_jaccard = dig_shared / len(s1_digs.union(c_digs))
        dig_exact = 1.0 if dig_shared > 0 else 0.0
    else:
        dig_shared = 0
        dig_jaccard = dig_exact = 0.0
        
    comb_tset = fuzz.token_set_ratio(s1_n_norm + " " + s1_a_norm, c_n_norm + " " + c_a_norm)
    c_match = 1.0 if str(s1_country).strip().upper() == str(c_country).strip().upper() else 0.0
    is_s2 = 1.0 if str(cid).startswith("S2-") else 0.0
    b_score = float(blocking_score)
    
    return [
        n_exact, n_exact, n_fuzz, n_part,
        n_tsort, n_tset, n_wratio, n_jaccard,
        n_len_diff, n_tok_diff, n_pre_sim, n_suf_sim,
        a_exact, a_fuzz, a_part, a_tsort,
        a_tset, a_jaccard, a_len_diff, addr_missing,
        dig_jaccard, dig_exact, float(dig_shared),
        comb_tset, c_match, is_s2, b_score
    ]

def clean_polars_df(df):
    return df.with_columns([
        pl.col("business_name").fill_null("").str.to_lowercase()
          .str.replace_all("&", " and ")
          .str.replace_all(r'[^\w\s]', " ")
          .str.replace_all(r'\s+', " ")
          .str.strip_chars()
          .alias("name_norm"),
          
        pl.col("business_address").fill_null("").str.to_lowercase()
          .str.replace_all(r'[^\w\s]', " ")
          .str.replace_all(r'\s+', " ")
          .str.strip_chars()
          .alias("addr_norm"),
          
        pl.col("country").fill_null("UNKNOWN").str.to_uppercase().str.strip_chars().alias("country_norm")
    ])

def main():
    print("=" * 70)
    print("STARTING FAST TEST SET SCORING & MATCHING RESULTS GENERATION")
    print("=" * 70)
    
    # 1. Train Model from cached training features
    feat_cache = os.path.join(MODELS_DIR, "train_features.npz")
    print(f"Loading training features from {feat_cache}...")
    data = np.load(feat_cache, allow_pickle=True)
    X_train = data["X_all"]
    y_train = data["y_all"]
    print(f"Training LightGBM on {len(X_train)} samples...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'is_unbalance': True,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.train(params, train_data, num_boost_round=300)
    print("LightGBM model training complete.")
    
    # 2. Ingest Test Data
    print("Loading test dataset entities...")
    t0 = time.time()
    df_s1 = pl.read_csv(os.path.join(DATASET_DIR, "test_source1.tsv"), separator="\t")
    df_s1 = clean_polars_df(df_s1)
    
    s1_dict = {}
    for r in df_s1.select(["entity_id", "name_norm", "addr_norm", "country_norm"]).iter_rows():
        s1_dict[r[0]] = (r[1], r[2], r[3])
    print(f"Loaded {len(s1_dict)} Source 1 test entities in {time.time()-t0:.2f}s")
    
    # 3. Read candidate pairs file to find needed candidates
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    print(f"Scanning needed candidates from {cand_path} (top 5 per S1)...")
    t0 = time.time()
    needed_cands = set()
    s1_top_cands = []
    
    with open(cand_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                c_list = [c.strip() for c in parts[1].split(",") if c.strip()][:5]
                s1_top_cands.append((s1_id, c_list))
                for c in c_list:
                    needed_cands.add(c)
            else:
                s1_top_cands.append((s1_id, []))
                
    print(f"Parsed {len(s1_top_cands)} S1 entities, identified {len(needed_cands)} unique candidates to score in {time.time()-t0:.2f}s")
    
    # 4. Load only needed candidates from S2 and S3
    t0 = time.time()
    df_s2 = pl.read_csv(os.path.join(DATASET_DIR, "test_source2.tsv"), separator="\t")
    df_s3 = pl.read_csv(os.path.join(DATASET_DIR, "test_source3.tsv"), separator="\t")
    s23_comb = pl.concat([df_s2, df_s3]).filter(pl.col("entity_id").is_in(list(needed_cands)))
    s23_clean = clean_polars_df(s23_comb)
    
    cand_dict = {}
    for r in s23_clean.select(["entity_id", "name_norm", "addr_norm", "country_norm"]).iter_rows():
        cand_dict[r[0]] = (r[1], r[2], r[3])
    print(f"Cached {len(cand_dict)} candidate records into memory in {time.time()-t0:.2f}s")
    
    # 5. Fast Batch Scoring
    print("Scoring candidate pairs with LightGBM (threshold = 0.60)...")
    t0 = time.time()
    THRESHOLD = 0.60
    BATCH_SIZE = 100000
    
    match_out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    
    with open(match_out_path, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        
        pair_buffer = [] # (s1_id, cand_id, features)
        pairs_scored = 0
        matches_found = 0
        
        for idx, (s1_id, c_list) in enumerate(s1_top_cands):
            s1_info = s1_dict.get(s1_id)
            if not s1_info or not c_list:
                out_f.write(f"{s1_id}\t\n")
                continue
                
            s1_n, s1_a, s1_c = s1_info
            
            # Extract features for all candidates of this S1
            s1_feats = []
            valid_cids = []
            for rank_i, cid in enumerate(c_list):
                c_info = cand_dict.get(cid)
                if not c_info:
                    continue
                c_n, c_a, c_c = c_info
                # Assign decaying blocking score by rank
                b_score = 5.0 / (1.0 + rank_i)
                feat = fast_pairwise_features(s1_n, s1_a, s1_c, cid, c_n, c_a, c_c, b_score)
                s1_feats.append(feat)
                valid_cids.append(cid)
                
            if s1_feats:
                probs = model.predict(np.array(s1_feats, dtype=np.float32))
                matched_cids = [cid for cid, p in zip(valid_cids, probs) if p >= THRESHOLD]
                pairs_scored += len(s1_feats)
                matches_found += len(matched_cids)
                out_f.write(f"{s1_id}\t{','.join(matched_cids)}\n")
            else:
                out_f.write(f"{s1_id}\t\n")
                
            if (idx + 1) % 200000 == 0:
                print(f"Processed {idx+1:,} / {len(s1_top_cands):,} S1 entities ({pairs_scored:,} pairs scored, {matches_found:,} matches) in {time.time()-t0:.2f}s")
                
    print(f"Finished scoring in {time.time()-t0:.2f}s! Total pairs scored: {pairs_scored:,}, Matches found: {matches_found:,}")
    print(f"Saved matching results to {match_out_path}")
    
    # 6. Validate with challenge validator
    validator_path = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        print("Running official submission validator...")
        import subprocess
        cmd = [
            sys.executable, validator_path,
            "--matching", match_out_path,
            "--candidate", cand_path,
            "--test-dir", DATASET_DIR
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("✅ SUBMISSION INTEGRITY CHECK: PASS (Exit code 0)")
        else:
            print(f"❌ Validator returned code {res.returncode}")

if __name__ == "__main__":
    main()
