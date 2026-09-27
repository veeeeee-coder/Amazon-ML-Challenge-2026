import numpy as np
from rapidfuzz import fuzz
from normalization import normalize_name, normalize_address, extract_numeric_tokens, extract_significant_tokens

FEATURE_NAMES = [
    # Name features
    "name_exact_match", "name_norm_equality", "name_fuzz_ratio", "name_partial_ratio",
    "name_token_sort", "name_token_set", "name_wratio", "name_jaccard",
    "name_len_diff", "name_token_count_diff", "name_prefix_sim", "name_suffix_sim",
    
    # Address features
    "addr_exact_match", "addr_fuzz_ratio", "addr_partial_ratio", "addr_token_sort",
    "addr_token_set", "addr_jaccard", "addr_len_diff", "addr_missing_flag",
    "dig_jaccard", "dig_exact_match", "dig_count_shared",
    
    # Joint & Structural features
    "comb_token_set", "country_exact_match", "is_s2", "blocking_score"
]

def extract_pairwise_features(
    s1_name_raw: str,
    s1_addr_raw: str,
    s1_country: str,
    cand_id: str,
    cand_name_raw: str,
    cand_addr_raw: str,
    cand_country: str,
    blocking_score: float = 0.0
) -> list:
    """Computes comprehensive 27-dimensional pairwise feature vector."""
    s1_n_raw = str(s1_name_raw or "")
    s1_a_raw = str(s1_addr_raw or "")
    c_n_raw = str(cand_name_raw or "")
    c_a_raw = str(cand_addr_raw or "")
    
    s1_n_norm = normalize_name(s1_n_raw)
    s1_a_norm = normalize_address(s1_a_raw)
    c_n_norm = normalize_name(c_n_raw)
    c_a_norm = normalize_address(c_a_raw)
    
    # ------------------ NAME FEATURES ------------------
    n_exact = 1.0 if s1_n_raw.lower() == c_n_raw.lower() and s1_n_raw != "" else 0.0
    n_norm_eq = 1.0 if s1_n_norm == c_n_norm and s1_n_norm != "" else 0.0
    
    n_fuzz = fuzz.ratio(s1_n_norm, c_n_norm)
    n_part = fuzz.partial_ratio(s1_n_norm, c_n_norm)
    n_tsort = fuzz.token_sort_ratio(s1_n_norm, c_n_norm)
    n_tset = fuzz.token_set_ratio(s1_n_norm, c_n_norm)
    n_wratio = fuzz.WRatio(s1_n_raw, c_n_raw)
    
    s1_n_toks = extract_significant_tokens(s1_n_norm)
    c_n_toks = extract_significant_tokens(c_n_norm)
    if s1_n_toks and c_n_toks:
        n_jaccard = len(s1_n_toks.intersection(c_n_toks)) / len(s1_n_toks.union(c_n_toks))
    else:
        n_jaccard = 0.0
        
    n_len_diff = abs(len(s1_n_norm) - len(c_n_norm))
    n_tok_diff = abs(len(s1_n_toks) - len(c_n_toks))
    
    # Prefix / suffix similarity (first / last 4 chars)
    s1_pre = s1_n_norm[:4] if len(s1_n_norm) >= 4 else s1_n_norm
    c_pre = c_n_norm[:4] if len(c_n_norm) >= 4 else c_n_norm
    n_pre_sim = 1.0 if s1_pre == c_pre and s1_pre != "" else 0.0
    
    s1_suf = s1_n_norm[-4:] if len(s1_n_norm) >= 4 else s1_n_norm
    c_suf = c_n_norm[-4:] if len(c_n_norm) >= 4 else c_n_norm
    n_suf_sim = 1.0 if s1_suf == c_suf and s1_suf != "" else 0.0

    # ------------------ ADDRESS FEATURES ------------------
    addr_missing = 1.0 if (not s1_a_raw or s1_a_raw == "None" or not c_a_raw or c_a_raw == "None") else 0.0
    
    if addr_missing == 0.0:
        a_exact = 1.0 if s1_a_raw.lower() == c_a_raw.lower() else 0.0
        a_fuzz = fuzz.ratio(s1_a_norm, c_a_norm)
        a_part = fuzz.partial_ratio(s1_a_norm, c_a_norm)
        a_tsort = fuzz.token_sort_ratio(s1_a_norm, c_a_norm)
        a_tset = fuzz.token_set_ratio(s1_a_norm, c_a_norm)
        
        s1_a_toks = extract_significant_tokens(s1_a_norm)
        c_a_toks = extract_significant_tokens(c_a_norm)
        if s1_a_toks and c_a_toks:
            a_jaccard = len(s1_a_toks.intersection(c_a_toks)) / len(s1_a_toks.union(c_a_toks))
        else:
            a_jaccard = 0.0
            
        a_len_diff = abs(len(s1_a_norm) - len(c_a_norm))
    else:
        a_exact = 0.0
        a_fuzz = 0.0
        a_part = 0.0
        a_tsort = 0.0
        a_tset = 0.0
        a_jaccard = 0.0
        a_len_diff = abs(len(s1_a_norm) - len(c_a_norm))
        
    # Digits matching
    s1_digs = extract_numeric_tokens(s1_n_raw + " " + s1_a_raw)
    c_digs = extract_numeric_tokens(c_n_raw + " " + c_a_raw)
    
    if s1_digs and c_digs:
        dig_shared = len(s1_digs.intersection(c_digs))
        dig_jaccard = dig_shared / len(s1_digs.union(c_digs))
        dig_exact = 1.0 if dig_shared > 0 else 0.0
    else:
        dig_shared = 0
        dig_jaccard = 0.0
        dig_exact = 0.0
        
    # ------------------ JOINT & STRUCTURAL ------------------
    comb_tset = fuzz.token_set_ratio(s1_n_norm + " " + s1_a_norm, c_n_norm + " " + c_a_norm)
    c_match = 1.0 if str(s1_country).strip().upper() == str(cand_country).strip().upper() else 0.0
    is_s2 = 1.0 if str(cand_id).startswith("S2-") else 0.0
    b_score = float(blocking_score) if blocking_score is not None and not np.isnan(blocking_score) else 0.0
    
    return [
        n_exact, n_norm_eq, n_fuzz, n_part,
        n_tsort, n_tset, n_wratio, n_jaccard,
        n_len_diff, n_tok_diff, n_pre_sim, n_suf_sim,
        
        a_exact, a_fuzz, a_part, a_tsort,
        a_tset, a_jaccard, a_len_diff, addr_missing,
        dig_jaccard, dig_exact, float(dig_shared),
        
        comb_tset, c_match, is_s2, b_score
    ]
