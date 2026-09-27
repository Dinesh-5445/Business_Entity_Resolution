def select_matches_for_s1(s1_id, candidate_scores, threshold=0.5, 
                          margin=0.1, min_candidates=1):
    """
    Makes the final match decision for a single S1 entity based on calibrated scores.
    
    Args:
        s1_id (str): The Source 1 entity ID.
        candidate_scores (list of tuple): List of (candidate_id, calibrated_score)
        threshold (float): Minimum score to be considered a match.
        margin (float): If the top score is not significantly higher than others, 
                        we might have collisions/uncertainty.
        min_candidates (int): Minimum candidates to predict if threshold is met.
        
    Returns:
        list: Selected candidate IDs.
    """
    if not candidate_scores:
        return []
        
    # Sort by score descending
    sorted_cands = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
    
    selected = []
    
    # 1. Singleton Safeguard
    # If the top score is below our threshold, predict EMPTY (singleton)
    if sorted_cands[0][1] < threshold:
        return []
        
    # 2. Multi-Match Decision
    # A candidate must independently provide sufficient evidence (>= threshold).
    for cand_id, score in sorted_cands:
        if score >= threshold:
            selected.append(cand_id)
            
    # Optionally, we could add collision suppression here
    # E.g., if we only want to predict multiple if scores are very close
    
    return selected

def generate_matching_results(s1_data, s1_to_candidates_scores, output_path, threshold=0.5):
    """
    Generates the official matching_results.tsv file.
    
    Args:
        s1_data (list of dict): S1 records.
        s1_to_candidates_scores (dict): {s1_id: [(cand_id, score), ...]}
        output_path (str): Path to write matching_results.tsv
        threshold (float): Decision threshold.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        
        for row in s1_data:
            s1_id = row['entity_id']
            cand_scores = s1_to_candidates_scores.get(s1_id, [])
            
            selected = select_matches_for_s1(s1_id, cand_scores, threshold=threshold)
            
            matched_str = ",".join(selected)
            f.write(f"{s1_id}\t{matched_str}\n")
