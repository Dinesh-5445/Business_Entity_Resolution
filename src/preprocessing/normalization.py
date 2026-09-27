import re
import unicodedata

# Common legal suffixes (basic set, can be expanded based on EDA)
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 
    'co', 'company', 'plc', 'gmbh', 'sa', 'nv', 'bv', 'srl', 'spa'
}

def remove_accents(text):
    if not text: return ""
    try:
        text = str(text)
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    except:
        return text

def normalize_text(text):
    """Base normalization: lowercase, strip, remove accents."""
    if not text or str(text).strip() == "":
        return ""
    text = str(text).lower()
    text = remove_accents(text)
    return text.strip()

def remove_punctuation(text):
    """Replaces punctuation with space."""
    return re.sub(r'[^\w\s]', ' ', text)

def alphanumeric_compact(text):
    """Removes all non-alphanumeric characters and spaces."""
    return re.sub(r'[^a-z0-9]', '', text)

def tokenize(text):
    """Returns list of alphanumeric tokens."""
    if not text: return []
    return text.split()

def remove_legal_suffixes(tokens):
    """Removes trailing legal suffixes."""
    if not tokens: return []
    res = []
    for t in tokens:
        if t not in LEGAL_SUFFIXES:
            res.append(t)
    return res

def get_numeric_tokens(tokens):
    """Extracts tokens that contain numbers."""
    return [t for t in tokens if re.search(r'\d', t)]

def extract_name_views(raw_name):
    """Generates multiple views for a name."""
    views = {}
    base = normalize_text(raw_name)
    views['raw_normalized'] = base
    
    no_punct = remove_punctuation(base)
    views['no_punct'] = " ".join(no_punct.split()) # normalize whitespace
    
    views['alphanumeric_compact'] = alphanumeric_compact(base)
    
    tokens = tokenize(views['no_punct'])
    views['tokens'] = tokens
    
    core_tokens = remove_legal_suffixes(tokens)
    views['core_tokens'] = core_tokens
    views['core_name'] = " ".join(core_tokens)
    
    return views

def extract_address_views(raw_address):
    """Generates multiple views for an address."""
    views = {}
    base = normalize_text(raw_address)
    views['raw_normalized'] = base
    
    no_punct = remove_punctuation(base)
    views['no_punct'] = " ".join(no_punct.split())
    
    views['alphanumeric_compact'] = alphanumeric_compact(base)
    
    tokens = tokenize(views['no_punct'])
    views['tokens'] = tokens
    
    views['numeric_tokens'] = get_numeric_tokens(tokens)
    
    # Token sorted
    views['token_sorted'] = " ".join(sorted(tokens))
    
    return views

def extract_country_views(raw_country):
    """Generates view for country."""
    return {'normalized': normalize_text(raw_country)}
