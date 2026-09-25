import re

STOPWORDS = {
    'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'pvt', 'private', 'co', 'company',
    'and', 'or', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'sa', 'sarl', 'gmbh',
    'enterprises', 'services', 'solutions', 'systems', 'traders', 'group', 'center', 'service',
    'shop', 'store', 'llp', 'inc.'
}

STOPWORDS_REGEX = r'\b(inc|llc|ltd|limited|corp|corporation|pvt|private|co|company|and|or|the|of|for|in|on|at|to|a|an|sa|sarl|gmbh|enterprises|services|solutions|systems|traders|group|center|service|shop|store|llp)\b'

ADDR_FILLERS_REGEX = r'\b(road|street|lane|nagar|city|flat|near|floor|block|plot|building|opp|opposite|behind|post|dist|district|state|maharashtra|delhi|karnataka|kerala|tamil|nadu|gujarat|bengal|uttar|pradesh|rajasthan|andhra|telangana|route|rue|avenue|bd|boulevard|cedex|paris)\b'

def clean_text(text: str) -> str:
    if not text or text == "None":
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_digits(text: str) -> set:
    if not text or text == "None":
        return set()
    return set(re.findall(r'\b\d{1,6}\b', str(text)))
