import argparse
import csv
import os
from pathlib import Path
import random

# Import components
from src.preprocessing.normalization import extract_name_views, extract_address_views, extract_country_views
from src.blocking.exact import ExactBlocker, exact_normalized_name_key
from src.retrieval.lexical import LexicalRetriever, extract_normalized_name_address
from src.blocking.union import union_candidates, generate_candidate_pairs_file
from src.features.pair_features import extract_pair_features
from src.models.lgb_matcher import LGBMatcher
from src.calibration.calibrator import PlattCalibrator
from src.decision.entity_decision import select_matches_for_s1, generate_matching_results

def load_data(filepath):
    print(f"Loading {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def preprocess_data(data):
    print("Preprocessing data...")
    processed = {}
    for row in data:
        eid = row['entity_id']
        views = {
            'name': extract_name_views(row.get('business_name', '')),
            'address': extract_address_views(row.get('business_address', '')),
            'country': extract_country_views(row.get('country', ''))
        }
        processed[eid] = views
    return processed

def run_pipeline(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load Data ---
    train_s1 = load_data(data_dir / 'train' / 'train_source1.tsv')
    train_s2 = load_data(data_dir / 'train' / 'train_source2.tsv')
    train_s3 = load_data(data_dir / 'train' / 'train_source3.tsv')
    
    # To save time for this initial pipeline run, we will sample data or assume it fits in memory
    # For a full run, we'd use all data
    
    # --- 2. Preprocess ---
    s1_views = preprocess_data(train_s1)
    s2_views = preprocess_data(train_s2)
    s3_views = preprocess_data(train_s3)
    
    all_target_views = {**s2_views, **s3_views}
    
    # --- 3. Candidate Generation ---
    print("Building Exact Blocker...")
    blocker = ExactBlocker(key_funcs=[exact_normalized_name_key])
    blocker.fit(train_s2, train_s3)
    
    print("Building Lexical Retriever...")
    retriever = LexicalRetriever(analyzer='char_wb', ngram_range=(3,4), max_features=50000)
    retriever.fit(train_s2, train_s3, text_extract_func=extract_normalized_name_address)
    
    print("Generating Candidates...")
    retriever_results = retriever.retrieve_batch(train_s1, extract_normalized_name_address, top_k=20, threshold=0.3)
    
    candidates_dict = {}
    for s1_row in train_s1:
        s1_id = s1_row['entity_id']
        
        cands_exact = blocker.retrieve(s1_row)
        cands_lexical = retriever_results.get(s1_id, [])
        
        # Channel 0: Exact, Channel 1: Lexical
        union_cands = union_candidates([cands_exact, cands_lexical], max_candidates=30)
        candidates_dict[s1_id] = union_cands
        
    # --- 4. Handoff to Stage B ---
    cand_pairs_path = output_dir / 'candidate_pairs.tsv'
    print(f"Writing {cand_pairs_path}...")
    generate_candidate_pairs_file(train_s1, candidates_dict, cand_pairs_path)
    
    # --- 5. Pair Feature Engine & Training Data Prep ---
    print("Loading Ground Truth...")
    gt_map = {}
    with open(data_dir / 'train' / 'train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        for row in reader:
            if not row: continue
            s1_id = row[0]
            targets = set([t.strip() for t in row[1].split(',')] if len(row) > 1 and row[1].strip() else [])
            gt_map[s1_id] = targets
            
    print("Extracting Features...")
    X = []
    y = []
    for s1_id, cands in candidates_dict.items():
        s1_v = s1_views[s1_id]
        true_matches = gt_map.get(s1_id, set())
        
        for cand in cands:
            cand_id = cand['entity_id']
            cand_v = all_target_views.get(cand_id)
            if not cand_v: continue
            
            feats = extract_pair_features(s1_v, cand_v, cand)
            X.append(feats)
            y.append(1 if cand_id in true_matches else 0)
            
    # --- 6. Train LightGBM ---
    print("Training LightGBM Matcher...")
    matcher = LGBMatcher()
    matcher.fit(X, y, num_boost_round=100, early_stopping_rounds=None) # MVP
    
    # --- 7. Calibration ---
    print("Training Calibrator...")
    # For MVP we calibrate on train scores (in practice use OOF)
    raw_scores = matcher.predict_proba(X)
    calibrator = PlattCalibrator()
    calibrator.fit(raw_scores, y)
    calibrated_scores = calibrator.predict_proba(raw_scores)
    
    # Re-map scores to candidates
    print("Scoring and making decisions...")
    idx = 0
    s1_to_scored_cands = {}
    for s1_id, cands in candidates_dict.items():
        scored = []
        for cand in cands:
            cand_id = cand['entity_id']
            if all_target_views.get(cand_id):
                score = calibrated_scores[idx]
                scored.append((cand_id, score))
                idx += 1
        s1_to_scored_cands[s1_id] = scored
        
    # --- 8. Entity-level decision ---
    match_results_path = output_dir / 'matching_results.tsv'
    print(f"Writing {match_results_path}...")
    generate_matching_results(train_s1, s1_to_scored_cands, match_results_path, threshold=0.5)
    
    print("Pipeline Complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()
    
    run_pipeline(args.data_dir, args.output_dir)
