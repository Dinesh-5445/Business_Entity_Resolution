import collections

def union_candidates(query_candidates_list, max_candidates=None):
    """
    Unions candidates from multiple retrieval channels for a single S1 query.
    
    Args:
        query_candidates_list (list of list): Each inner list is a list of candidate tuples 
            from a specific channel. E.g., [[(s2_1, score1), ...], [(s3_1, score2), ...]]
        max_candidates (int): Adaptive candidate cap.
        
    Returns:
        list of dict: Candidate provenance list.
    """
    candidate_map = collections.defaultdict(dict)
    
    for channel_idx, channel_cands in enumerate(query_candidates_list):
        for rank, cand_data in enumerate(channel_cands):
            if isinstance(cand_data, tuple) and len(cand_data) == 2:
                cand_id, score = cand_data
            else:
                cand_id = cand_data
                score = None
                
            if cand_id not in candidate_map:
                candidate_map[cand_id] = {
                    'entity_id': cand_id,
                    'channels': set(),
                    'scores': {},
                    'ranks': {}
                }
                
            candidate_map[cand_id]['channels'].add(channel_idx)
            if score is not None:
                candidate_map[cand_id]['scores'][channel_idx] = score
            candidate_map[cand_id]['ranks'][channel_idx] = rank
            
    # For adaptive candidate control, we can sort candidates by their highest score or number of channels
    # A simple robust heuristic: sort by (number of supporting channels, max score across channels)
    sorted_candidates = []
    for cand_id, data in candidate_map.items():
        max_score = max(data['scores'].values()) if data['scores'] else 0.0
        num_channels = len(data['channels'])
        sorted_candidates.append((num_channels, max_score, cand_id, data))
        
    sorted_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    if max_candidates is not None:
        sorted_candidates = sorted_candidates[:max_candidates]
        
    final_candidates = [x[3] for x in sorted_candidates]
    return final_candidates

def generate_candidate_pairs_file(s1_data, candidates_dict, output_path):
    """
    Generates the official candidate_pairs.tsv file.
    
    Args:
        s1_data (list of dict): S1 records.
        candidates_dict (dict): {s1_id: [cand_data_dict_1, cand_data_dict_2, ...]}
        output_path (str): Path to write candidate_pairs.tsv.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_data:
            s1_id = s1['entity_id']
            cands = candidates_dict.get(s1_id, [])
            cand_ids = [c['entity_id'] for c in cands]
            cand_str = ",".join(cand_ids)
            f.write(f"{s1_id}\t{cand_str}\n")
