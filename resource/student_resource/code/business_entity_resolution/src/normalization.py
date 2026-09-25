import re
import unicodedata
from typing import Set, List
import polars as pl

# Legal suffixes and common company designations across jurisdictions (US, India, France, International)
LEGAL_SUFFIXES_REGEX = r'\b(private\s+limited|pvt\s+ltd|pvt\s+limited|p\s+ltd|p\s+limited|llp|llc|inc|incorporated|corp|corporation|ltd|limited|co|company|gmbh|sa|sarl|sas|sasu|eurl|spa|bv|nv|traders|enterprises|services|solutions|systems|group|holdings|technologies|associates|partners|center|shop|store)\b'

# Address noise / filler terms
ADDR_FILLERS_REGEX = r'\b(road|street|lane|nagar|city|flat|near|floor|block|plot|building|opp|opposite|behind|post|dist|district|state|route|rue|avenue|bd|boulevard|cedex|all[eé]e|impasse|place|chemin)\b'

def unicode_normalize(text: str) -> str:
    """Applies NFKD Unicode normalization and ASCII transliteration."""
    if not text or text == "None":
        return ""
    text = unicodedata.normalize('NFKD', str(text))
    return text.encode('ascii', 'ignore').decode('utf-8')

def normalize_name(text: str) -> str:
    """
    Normalizes business name:
    - Lowercase & Unicode normalization
    - Ampersand expansion (& -> and)
    - Punctuation removal
    - Standard legal suffix normalization
    - Whitespace trimming
    """
    if not text or text == "None":
        return ""
    text = unicode_normalize(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " ")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(LEGAL_SUFFIXES_REGEX, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_address(text: str) -> str:
    """
    Normalizes business address:
    - Lowercase & Unicode normalization
    - Standard address abbreviations (st -> street, rd -> road, ave -> avenue, etc.)
    - Address filler word cleaning
    - Whitespace cleanup
    """
    if not text or text == "None":
        return ""
    text = unicode_normalize(text).lower()
    text = re.sub(r'\b(rd|rd\.)\b', 'road', text)
    text = re.sub(r'\b(st|st\.)\b', 'street', text)
    text = re.sub(r'\b(ave|ave\.)\b', 'avenue', text)
    text = re.sub(r'\b(ln|ln\.)\b', 'lane', text)
    text = re.sub(r'\b(blvd|blvd\.)\b', 'boulevard', text)
    text = re.sub(r'\b(apt|apt\.)\b', 'apartment', text)
    text = re.sub(r'\b(no|no\.)\b', 'number', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_country(text: str) -> str:
    """Normalizes country field without hard-coding specific countries."""
    if not text or text == "None":
        return "UNKNOWN"
    return unicode_normalize(text).strip().upper()

def extract_numeric_tokens(text: str) -> Set[str]:
    """Extracts all house numbers, street numbers, PIN/zip codes from text."""
    if not text or text == "None":
        return set()
    return set(re.findall(r'\b\d{1,6}\b', str(text)))

def extract_significant_tokens(text: str, min_len: int = 3) -> Set[str]:
    """Extracts word tokens with length >= min_len."""
    if not text or text == "None":
        return set()
    cleaned = normalize_name(text)
    return set(re.findall(rf'\b[a-z0-9]{{{min_len},}}\b', cleaned))
