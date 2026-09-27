import os
import sys
import re
import unicodedata
from typing import Optional, List, Dict, Any
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: Missing web hosting dependencies ('fastapi' and/or 'uvicorn').")
    print("Please install them by running:")
    print("    pip install fastapi uvicorn")
    print("or install all project requirements:")
    print("    pip install -r requirements.txt")
    print("=" * 70 + "\n")
    sys.exit(1)

# ---------------------------------------------------------
# Dynamic Path Resolution & Internal Module Loading
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Attempt to load project normalization & fuzzy tools, with robust fallbacks
try:
    from normalization import normalize_name, normalize_address, extract_numeric_tokens
except ImportError:
    LEGAL_SUFFIXES_REGEX = r'\b(private\s+limited|pvt\s+ltd|pvt\s+limited|p\s+ltd|p\s+limited|llp|llc|inc|incorporated|corp|corporation|ltd|limited|co|company|gmbh|sa|sarl|sas|sasu|eurl|spa|bv|nv|traders|enterprises|services|solutions|systems|group|holdings|technologies|associates|partners|center|shop|store)\b'
    
    def unicode_normalize(text: str) -> str:
        if not text or text == "None":
            return ""
        text = unicodedata.normalize('NFKD', str(text))
        return text.encode('ascii', 'ignore').decode('utf-8')

    def normalize_name(text: str) -> str:
        if not text or text == "None":
            return ""
        text = unicode_normalize(text).lower()
        text = text.replace("&", " and ")
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(LEGAL_SUFFIXES_REGEX, ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_address(text: str) -> str:
        if not text or text == "None":
            return ""
        text = unicode_normalize(text).lower()
        text = re.sub(r'\b(rd|rd\.)\b', 'road', text)
        text = re.sub(r'\b(st|st\.)\b', 'street', text)
        text = re.sub(r'\b(ave|ave\.)\b', 'avenue', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_numeric_tokens(text: str) -> set:
        if not text:
            return set()
        return set(re.findall(r'\b\d{1,6}\b', str(text)))

# Attempt RapidFuzz import with pure Python fallback
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

def compute_token_set_ratio(s1: str, s2: str) -> float:
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(s1, s2) / 100.0
    toks1 = set(s1.split())
    toks2 = set(s2.split())
    if not toks1 or not toks2:
        return 0.0
    return len(toks1.intersection(toks2)) / len(toks1.union(toks2))

def compute_token_sort_ratio(s1: str, s2: str) -> float:
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    s1_sorted = " ".join(sorted(s1.split()))
    s2_sorted = " ".join(sorted(s2.split()))
    if s1_sorted == s2_sorted and s1_sorted != "":
        return 1.0
    toks1 = set(s1.split())
    toks2 = set(s2.split())
    return (2.0 * len(toks1.intersection(toks2))) / (len(toks1) + len(toks2)) if (toks1 or toks2) else 0.0

# ---------------------------------------------------------
# FastAPI Application Definition & CORS Middleware
# ---------------------------------------------------------
app = FastAPI(
    title="Amazon ML Challenge: Business Entity Resolution API",
    description="Production REST API for high-precision Business Entity Matching across noisy records.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------
class EntityPairRequest(BaseModel):
    s1_name: str = Field(..., alias="source1_name", example="Dynamic Construction Pvt Ltd")
    s1_address: Optional[str] = Field("", alias="source1_address", example="Flat 102, Ratna Residency, Vuyyuru, Krishna, Andhra Pradesh")
    s1_country: Optional[str] = Field("India", alias="source1_country", example="India")
    
    cand_name: str = Field(..., alias="candidate_name", example="Dynamic Construction Limited")
    cand_address: Optional[str] = Field("", alias="candidate_address", example="Flat 123, Ratna Residency, Vuyyuru, Krishna, AP")
    cand_country: Optional[str] = Field("India", alias="candidate_country", example="India")
    cand_id: Optional[str] = Field("S2-001", alias="candidate_id", example="S2-001")

    class Config:
        allow_population_by_field_name = True

class MatchResponse(BaseModel):
    decision: str
    is_match: bool
    confidence_score: float
    threshold: float
    breakdown: Dict[str, Any]

class BatchCandidate(BaseModel):
    cand_id: str = Field(..., example="S2-1001")
    cand_name: str = Field(..., example="Dynamic Builders Ltd")
    cand_address: Optional[str] = Field("", example="Vuyyuru, Krishna")
    cand_country: Optional[str] = Field("India", example="India")

class BatchMatchRequest(BaseModel):
    s1_name: str = Field(..., example="Dynamic Construction Pvt Ltd")
    s1_address: Optional[str] = Field("", example="Flat 102, Ratna Residency, Vuyyuru, Krishna, Andhra Pradesh")
    s1_country: Optional[str] = Field("India", example="India")
    candidates: List[BatchCandidate]

class BatchMatchResponse(BaseModel):
    source1_entity: str
    total_candidates: int
    matches_found: int
    results: List[Dict[str, Any]]

# ---------------------------------------------------------
# Core Matching Logic
# ---------------------------------------------------------
def score_pair(s1_name: str, s1_addr: str, s1_country: str,
               c_name: str, c_addr: str, c_country: str) -> Dict[str, Any]:
    """Computes similarity and prediction between a reference entity and a candidate."""
    # 1. Country boundary check
    s1_c = (s1_country or "").strip().lower()
    c_c = (c_country or "").strip().lower()
    if s1_c and c_c and s1_c != c_c:
        return {
            "decision": "NO_MATCH",
            "is_match": False,
            "confidence_score": 0.0,
            "threshold": 0.60,
            "breakdown": {"reason": f"Cross-country partition barrier ({s1_c} != {c_c})"}
        }

    # 2. Text Normalization
    s1_n_norm = normalize_name(s1_name)
    c_n_norm = normalize_name(c_name)
    s1_a_norm = normalize_address(s1_addr or "")
    c_a_norm = normalize_address(c_addr or "")

    # 3. String Similarities
    name_set = compute_token_set_ratio(s1_n_norm, c_n_norm)
    name_sort = compute_token_sort_ratio(s1_n_norm, c_n_norm)
    
    # 4. Address & Numeric Match
    addr_present = bool(s1_a_norm and c_a_norm)
    addr_set = compute_token_set_ratio(s1_a_norm, c_a_norm) if addr_present else 0.5

    s1_digs = extract_numeric_tokens(s1_name + " " + (s1_addr or ""))
    c_digs = extract_numeric_tokens(c_name + " " + (c_addr or ""))
    dig_overlap = len(s1_digs.intersection(c_digs)) > 0 if (s1_digs and c_digs) else False

    # 5. Composite Confidence Score (Calibrated for Macro F0.5)
    if not addr_present:
        raw_score = 0.70 * name_set + 0.30 * name_sort
    else:
        raw_score = 0.50 * name_set + 0.20 * name_sort + 0.30 * addr_set

    # Boost slightly if numeric street/PIN anchors match exactly
    if dig_overlap:
        raw_score = min(1.0, raw_score + 0.05)

    confidence = round(raw_score, 4)
    threshold = 0.60
    is_match = bool(confidence >= threshold)

    return {
        "decision": "MATCH" if is_match else "NO_MATCH",
        "is_match": is_match,
        "confidence_score": confidence,
        "threshold": threshold,
        "breakdown": {
            "name_token_set_similarity": round(name_set, 4),
            "name_token_sort_similarity": round(name_sort, 4),
            "address_similarity": round(addr_set, 4) if addr_present else None,
            "numeric_anchor_match": dig_overlap,
            "s1_normalized_name": s1_n_norm,
            "candidate_normalized_name": c_n_norm
        }
    }

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Amazon Business Entity Resolution Engine",
        "status": "online",
        "optimal_threshold": 0.60,
        "metric_target": "Macro F0.5",
        "endpoints": {
            "swagger_docs": "/docs",
            "health_check": "/health",
            "single_pair_predict": "/predict",
            "batch_match": "/batch_predict"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Entity Matcher API"}

@app.post("/predict", response_model=MatchResponse)
def predict_match(pair: EntityPairRequest):
    """Predicts whether a single candidate record matches the Source 1 reference entity."""
    res = score_pair(
        s1_name=pair.s1_name,
        s1_addr=pair.s1_address or "",
        s1_country=pair.s1_country or "",
        c_name=pair.cand_name,
        c_addr=pair.cand_address or "",
        c_country=pair.cand_country or ""
    )
    return MatchResponse(**res)

@app.post("/batch_predict", response_model=BatchMatchResponse)
def batch_predict(request: BatchMatchRequest):
    """Evaluates and ranks a list of candidates against a single Source 1 reference entity."""
    results = []
    matches = 0
    for cand in request.candidates:
        scored = score_pair(
            s1_name=request.s1_name,
            s1_addr=request.s1_address or "",
            s1_country=request.s1_country or "",
            c_name=cand.cand_name,
            c_addr=cand.cand_address or "",
            c_country=cand.cand_country or ""
        )
        scored["cand_id"] = cand.cand_id
        scored["cand_name"] = cand.cand_name
        if scored["is_match"]:
            matches += 1
        results.append(scored)

    # Sort descending by confidence score
    results.sort(key=lambda x: x["confidence_score"], reverse=True)

    return BatchMatchResponse(
        source1_entity=request.s1_name,
        total_candidates=len(request.candidates),
        matches_found=matches,
        results=results
    )

# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Pass app instance directly so uvicorn runs reliably regardless of working directory
    uvicorn.run(app, host="0.0.0.0", port=8000)
