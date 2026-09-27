import difflib

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def char_similarity(str1, str2):
    if not str1 and not str2:
        return 0.0
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def extract_pair_features(s1_views, cand_views, cand_metadata):
    """
    Extracts features for a candidate pair.
    
    Args:
        s1_views (dict): Features/views of S1 record.
        cand_views (dict): Features/views of Candidate record.
        cand_metadata (dict): Context/provenance of candidate.
        
    Returns:
        list of floats: Feature vector.
    """
    features = []
    
    # Context features
    # cand_metadata contains: channels, scores, ranks
    channels = cand_metadata.get('channels', set())
    num_channels = len(channels)
    max_score = max(cand_metadata.get('scores', {0: 0.0}).values()) if cand_metadata.get('scores') else 0.0
    min_rank = min(cand_metadata.get('ranks', {0: 999}).values()) if cand_metadata.get('ranks') else 999
    
    features.append(float(num_channels))
    features.append(float(max_score))
    features.append(float(min_rank))
    
    # Missingness
    s1_name_missing = 1.0 if not s1_views['name']['raw_normalized'] else 0.0
    cand_name_missing = 1.0 if not cand_views['name']['raw_normalized'] else 0.0
    s1_addr_missing = 1.0 if not s1_views['address']['raw_normalized'] else 0.0
    cand_addr_missing = 1.0 if not cand_views['address']['raw_normalized'] else 0.0
    
    features.extend([s1_name_missing, cand_name_missing, s1_addr_missing, cand_addr_missing])
    
    # Name Features
    s1_name_tokens = set(s1_views['name'].get('tokens', []))
    cand_name_tokens = set(cand_views['name'].get('tokens', []))
    name_jaccard = jaccard_similarity(s1_name_tokens, cand_name_tokens)
    name_char_sim = char_similarity(
        s1_views['name'].get('alphanumeric_compact', ''), 
        cand_views['name'].get('alphanumeric_compact', '')
    )
    name_exact = 1.0 if s1_views['name']['raw_normalized'] == cand_views['name']['raw_normalized'] and not s1_name_missing else 0.0
    core_name_exact = 1.0 if s1_views['name'].get('core_name') == cand_views['name'].get('core_name') and not s1_name_missing else 0.0
    
    features.extend([name_jaccard, name_char_sim, name_exact, core_name_exact])
    
    # Address Features
    s1_addr_tokens = set(s1_views['address'].get('tokens', []))
    cand_addr_tokens = set(cand_views['address'].get('tokens', []))
    addr_jaccard = jaccard_similarity(s1_addr_tokens, cand_addr_tokens)
    addr_char_sim = char_similarity(
        s1_views['address'].get('alphanumeric_compact', ''),
        cand_views['address'].get('alphanumeric_compact', '')
    )
    
    s1_num_tokens = set(s1_views['address'].get('numeric_tokens', []))
    cand_num_tokens = set(cand_views['address'].get('numeric_tokens', []))
    num_jaccard = jaccard_similarity(s1_num_tokens, cand_num_tokens)
    num_exact = 1.0 if s1_num_tokens and s1_num_tokens == cand_num_tokens else 0.0
    num_contradiction = 1.0 if s1_num_tokens and cand_num_tokens and len(s1_num_tokens.intersection(cand_num_tokens)) == 0 else 0.0
    
    features.extend([addr_jaccard, addr_char_sim, num_jaccard, num_exact, num_contradiction])
    
    # Country agreement
    s1_country = s1_views['country']['normalized']
    cand_country = cand_views['country']['normalized']
    country_missing = 1.0 if not s1_country or not cand_country else 0.0
    country_agree = 1.0 if s1_country and s1_country == cand_country else 0.0
    country_contradict = 1.0 if s1_country and cand_country and s1_country != cand_country else 0.0
    
    features.extend([country_missing, country_agree, country_contradict])
    
    return features

def get_feature_names():
    return [
        'num_channels', 'max_score', 'min_rank',
        's1_name_missing', 'cand_name_missing', 's1_addr_missing', 'cand_addr_missing',
        'name_jaccard', 'name_char_sim', 'name_exact', 'core_name_exact',
        'addr_jaccard', 'addr_char_sim', 'num_jaccard', 'num_exact', 'num_contradiction',
        'country_missing', 'country_agree', 'country_contradict'
    ]
