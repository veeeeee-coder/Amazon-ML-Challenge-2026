import numpy as np
from typing import Dict, Set, List, Tuple
from collections import defaultdict
from utils import logger

class EntityMatcher:
    """
    Inference Matcher that produces final entity resolution matches:
    - Supports 1 -> 0 (singletons), 1 -> 1, and 1 -> many matching
    - Enforces match consistency rules:
      * No self-matches (S1-* ids rejected)
      * Match IDs must have valid S2-/S3- prefix
      * Predicted matches must be a strict subset of candidates
      * No duplicate entity IDs within match lists
    """
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def match_candidates(
        self,
        pair_ids: List[Tuple[str, str]],
        probabilities: np.ndarray,
        all_s1_ids: List[str],
        candidate_dict: Dict[str, Set[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Applies calibrated decision threshold and consistency rules to generate match mapping.
        """
        match_dict = defaultdict(list)
        
        for idx, (s1_id, cand_id) in enumerate(pair_ids):
            prob = probabilities[idx]
            
            # Reject self-matches & invalid prefixes
            if cand_id.startswith("S1-") or not (cand_id.startswith("S2-") or cand_id.startswith("S3-")):
                continue
                
            # If candidate_dict is provided, enforce subset constraint
            if candidate_dict is not None and cand_id not in candidate_dict.get(s1_id, set()):
                continue
                
            if prob >= self.threshold:
                match_dict[s1_id].append(cand_id)
                
        # Final formatting: ensure all S1 entities exist, deduplicate matches
        final_predictions = {}
        for s1_id in all_s1_ids:
            raw_matches = match_dict.get(s1_id, [])
            # Deduplicate preserving order
            unique_matches = list(dict.fromkeys(raw_matches))
            final_predictions[s1_id] = unique_matches
            
        return final_predictions
