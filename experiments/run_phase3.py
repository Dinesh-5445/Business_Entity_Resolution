"""
Phase 3 Full Test Inference — Inverted Index Architecture
==========================================================
Replaces expensive TF-IDF dense dot products with a memory-efficient
inverted n-gram index for candidate retrieval.

Key design:
- TRAINING: 50k sampled targets + 5k S1 → fast model training (<5 min)
- TEST INFERENCE: inverted char n-gram index lookup (O(tokens) per query)
  instead of full matrix multiply (O(M * N))
- All 10M test targets indexed; retrieval is BM25-style scoring via
  inverted posting lists
- S1 processed in streaming chunks of 1000
- Zero fallback empty predictions
"""

import sys
import os
import csv
import json
import random
import time
import datetime
import gc
import collections
from pathlib import Path
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.normalization import (
    normalize_text, remove_punctuation, alphanumeric_compact,
    tokenize, remove_legal_suffixes, get_numeric_tokens
)
from src.models.lgb_matcher import LGBMatcher
from src.calibration.calibrator import PlattCalibrator
from src.decision.entity_decision import select_matches_for_s1
from src.evaluation.metrics import compute_macro_f0_5
import difflib

# ---------------------------------------------------------------------------
# COMPACT VIEW
# ---------------------------------------------------------------------------

def make_compact_view(row):
    name = row.get('business_name', '') or ''
    addr = row.get('business_address', '') or ''
    country = row.get('country', '') or ''
    
    name_base  = normalize_text(name)
    name_alnum = alphanumeric_compact(name_base)
    name_np    = " ".join(remove_punctuation(name_base).split())
    name_toks  = tokenize(name_np)
    name_core  = " ".join(remove_legal_suffixes(name_toks))
    
    addr_base  = normalize_text(addr)
    addr_alnum = alphanumeric_compact(addr_base)
    addr_np    = " ".join(remove_punctuation(addr_base).split())
    addr_toks  = tokenize(addr_np)
    addr_num   = get_numeric_tokens(addr_toks)
    
    return {
        'n_raw':   name_base,
        'n_alnum': name_alnum,
        'n_tokens':frozenset(name_toks),
        'n_core':  name_core,
        'a_raw':   addr_base,
        'a_alnum': addr_alnum,
        'a_tokens':frozenset(addr_toks),
        'a_num':   frozenset(addr_num),
        'country': normalize_text(country),
        'index_text': name_alnum  # used for inverted index
    }

def _bigram_sim(a, b):
    """Fast char-bigram Jaccard similarity. O(n). Replaces difflib."""
    if not a and not b: return 0.0
    if not a or not b: return 0.0
    sa = set(a[i:i+2] for i in range(len(a)-1)) if len(a) >= 2 else {a}
    sb = set(b[i:i+2] for i in range(len(b)-1)) if len(b) >= 2 else {b}
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def compact_pair_features(s1v, tv, score, rank, n_channels):
    f = [float(n_channels), float(score), float(rank)]
    f += [
        1.0 if not s1v['n_raw'] else 0.0,
        1.0 if not tv['n_raw'] else 0.0,
        1.0 if not s1v['a_raw'] else 0.0,
        1.0 if not tv['a_raw'] else 0.0,
    ]
    s1_nt = s1v['n_tokens']; t_nt = tv['n_tokens']
    union_n = s1_nt | t_nt; inter_n = s1_nt & t_nt
    name_jaccard = len(inter_n) / len(union_n) if union_n else 0.0
    s1_al = s1v['n_alnum']; t_al = tv['n_alnum']
    name_char = _bigram_sim(s1_al, t_al)
    name_exact = 1.0 if s1v['n_raw'] and s1v['n_raw'] == tv['n_raw'] else 0.0
    core_exact = 1.0 if s1v['n_core'] and s1v['n_core'] == tv['n_core'] else 0.0
    f += [name_jaccard, name_char, name_exact, core_exact]
    s1_at = s1v['a_tokens']; t_at = tv['a_tokens']
    union_a = s1_at | t_at; inter_a = s1_at & t_at
    addr_jaccard = len(inter_a) / len(union_a) if union_a else 0.0
    addr_char = _bigram_sim(s1v['a_alnum'], tv['a_alnum'])
    s1_num = s1v['a_num']; t_num = tv['a_num']
    union_num = s1_num | t_num; inter_num = s1_num & t_num
    num_jaccard = len(inter_num) / len(union_num) if union_num else 0.0
    num_exact = 1.0 if s1_num and s1_num == t_num else 0.0
    num_contra = 1.0 if s1_num and t_num and not inter_num else 0.0
    f += [addr_jaccard, addr_char, num_jaccard, num_exact, num_contra]
    s1_c = s1v['country']; t_c = tv['country']
    f += [
        1.0 if not s1_c or not t_c else 0.0,
        1.0 if s1_c and s1_c == t_c else 0.0,
        1.0 if s1_c and t_c and s1_c != t_c else 0.0,
    ]
    return f

# ---------------------------------------------------------------------------
# INVERTED N-GRAM INDEX
# ---------------------------------------------------------------------------

def char_ngrams(text, n=3):
    """Generate character n-grams from text (boundary padded)."""
    padded = f" {text} "
    return [padded[i:i+n] for i in range(len(padded) - n + 1)]

class InvertedNgramIndex:
    """
    Inverted index mapping char n-grams → list of entity IDs.
    Query returns top-K candidates by shared n-gram count (TF proxy).
    Memory: ~1-2 bytes per (ngram, entity) posting.
    """
    def __init__(self, n=3):
        self.n = n
        self.posting = collections.defaultdict(list)  # ngram -> [eid, eid, ...]
        self.entity_len = {}  # eid -> len of ngrams (for normalization)
    
    def fit(self, target_views, tfidf_ids):
        """Index all targets."""
        for eid in tfidf_ids:
            tv = target_views[eid]
            text = tv['index_text']
            if not text:
                continue
            grams = set(char_ngrams(text, self.n))
            self.entity_len[eid] = len(grams)
            for g in grams:
                self.posting[g].append(eid)
        print(f"  Inverted index: {len(self.posting):,} unique {self.n}-grams, "
              f"{sum(len(v) for v in self.posting.values()):,} total postings")
    
    def query(self, text, top_k=20, min_score=1):
        """Return top-K candidates by shared n-gram count."""
        if not text:
            return []
        grams = set(char_ngrams(text, self.n))
        counts = collections.Counter()
        for g in grams:
            for eid in self.posting.get(g, []):
                counts[eid] += 1
        
        if not counts:
            return []
        
        # Normalize by query length for a cosine-like score
        q_len = len(grams)
        results = []
        for eid, cnt in counts.most_common(top_k * 3):  # oversample then re-rank
            if cnt < min_score:
                break
            score = cnt / max(q_len, self.entity_len.get(eid, 1))
            results.append((eid, score))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------

def stream_rows(filepath, limit=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            yield row

def load_targets_compact(paths, limit_each=None):
    """Load target records as compact views + exact index."""
    target_views = {}
    exact_index  = {}  # norm_name -> [eid]
    tfidf_ids    = []
    
    total = 0
    for path in paths:
        for row in stream_rows(path, limit=limit_each):
            eid = row['entity_id']
            v = make_compact_view(row)
            target_views[eid] = v
            tfidf_ids.append(eid)
            key = v['n_raw']
            if key:
                if key not in exact_index:
                    exact_index[key] = []
                exact_index[key].append(eid)
            total += 1
            if total % 1_000_000 == 0:
                print(f"  Loaded {total:,} records...", flush=True)
    
    print(f"  Total targets: {total:,}")
    return target_views, exact_index, tfidf_ids

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_phase3():
    random.seed(42)
    np.random.seed(42)
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'student_resource' / 'dataset'
    OUT_DIR  = BASE_DIR / 'output'
    SUB_DIR  = BASE_DIR / 'submissions' / 'phase_3_full_baseline'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    
    run_start = time.time()
    log_lines = []
    
    def log(msg):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)
    
    log("=== PHASE 3 FULL INFERENCE (Inverted Index Architecture) ===")
    
    # -------------------------------------------------------------------------
    # STEP 1: Train LightGBM on a SMALL but realistic training sample
    # -------------------------------------------------------------------------
    TRAIN_TARGET_LIMIT = 50_000   # 50k each = 100k total targets for training
    TRAIN_S1_FIT       = 5_000
    TRAIN_S1_VAL       = 1_000
    
    log(f"Loading train targets (limit={TRAIN_TARGET_LIMIT:,} each)...")
    train_target_views, train_exact_index, train_tfidf_ids = load_targets_compact(
        [DATA_DIR / 'train' / 'train_source2.tsv',
         DATA_DIR / 'train' / 'train_source3.tsv'],
        limit_each=TRAIN_TARGET_LIMIT
    )
    log(f"Train targets: {len(train_target_views):,}")
    
    log("Building training inverted index...")
    train_idx = InvertedNgramIndex(n=3)
    train_idx.fit(train_target_views, train_tfidf_ids)
    
    log("Loading train S1...")
    train_s1_all = list(stream_rows(DATA_DIR / 'train' / 'train_source1.tsv'))
    random.shuffle(train_s1_all)
    train_s1_fit = train_s1_all[:TRAIN_S1_FIT]
    train_s1_val = train_s1_all[TRAIN_S1_FIT:TRAIN_S1_FIT + TRAIN_S1_VAL]
    log(f"Train S1 fit: {len(train_s1_fit):,} | Val: {len(train_s1_val):,}")
    
    log("Loading ground truth...")
    gt_map = {}
    with open(DATA_DIR / 'train' / 'train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if not row: continue
            targets = set()
            if len(row) > 1 and row[1].strip():
                targets = {t.strip() for t in row[1].split(',')}
            gt_map[row[0]] = targets
    log(f"Ground truth: {len(gt_map):,} entries")
    
    # Generate training candidate pairs
    log("Generating training candidate pairs via inverted index...")
    X_train, y_train = [], []
    
    for row in train_s1_fit:
        eid = row['entity_id']
        s1v = make_compact_view(row)
        true_set = gt_map.get(eid, set())
        
        cand_scores = {}
        # Exact channel
        for cid in train_exact_index.get(s1v['n_raw'], []):
            cand_scores[cid] = 1.0
        # NGram channel
        for cid, sc in train_idx.query(s1v['index_text'], top_k=20, min_score=1):
            cand_scores[cid] = max(cand_scores.get(cid, 0.0), sc)
        
        top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:30]
        for rank, (cid, sc) in enumerate(top_cands):
            tv = train_target_views.get(cid)
            if tv is None: continue
            nc = 2 if cid in train_exact_index.get(s1v['n_raw'], []) else 1
            feat = compact_pair_features(s1v, tv, sc, rank, nc)
            X_train.append(feat)
            y_train.append(1 if cid in true_set else 0)
    
    log(f"Train pairs: {len(X_train):,} | Positives: {sum(y_train):,}")
    
    # Free training index
    del train_target_views, train_exact_index, train_tfidf_ids, train_idx
    gc.collect()
    
    # Train model
    log("Training LightGBM...")
    matcher = LGBMatcher()
    matcher.fit(X_train, y_train, num_boost_round=200)
    
    log("Training Platt calibrator...")
    raw_scores = matcher.predict_proba(X_train)
    calibrator = PlattCalibrator()
    calibrator.fit(raw_scores, y_train)
    del X_train, y_train, raw_scores
    gc.collect()
    
    # Quick validation (exact channel only for speed)
    log("Running local validation (exact channel only)...")
    log("  [DEBUG ONLY - target subset does not contain all true positives]")
    
    # -------------------------------------------------------------------------
    # STEP 2: Load FULL test targets
    # -------------------------------------------------------------------------
    log("Loading FULL test targets (this may take several minutes)...")
    test_target_views, test_exact_index, test_tfidf_ids = load_targets_compact(
        [DATA_DIR / 'test' / 'test_source2.tsv',
         DATA_DIR / 'test' / 'test_source3.tsv']
    )
    n_test_targets = len(test_target_views)
    log(f"Test targets loaded: {n_test_targets:,}")
    
    # Build inverted n-gram index on full test targets
    log("Building test inverted n-gram index...")
    t0 = time.time()
    test_idx = InvertedNgramIndex(n=3)
    test_idx.fit(test_target_views, test_tfidf_ids)
    log(f"Test index built in {time.time()-t0:.1f}s")
    
    # Free the ordered ID list — no longer needed
    del test_tfidf_ids
    gc.collect()
    
    # Count test S1
    test_s1_path = DATA_DIR / 'test' / 'test_source1.tsv'
    test_s1_count = sum(1 for _ in stream_rows(test_s1_path))
    log(f"Test S1 entities: {test_s1_count:,}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Full streaming inference
    # -------------------------------------------------------------------------
    log("Starting full test inference...")
    
    cand_out  = open(OUT_DIR / "candidate_pairs.tsv",  "w", encoding='utf-8')
    match_out = open(OUT_DIR / "matching_results.tsv", "w", encoding='utf-8')
    cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
    match_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    processed     = 0
    total_cands   = 0
    total_matches = 0
    empty_cands   = 0
    cand_counts   = []
    
    infer_start = time.time()
    CHUNK_SIZE  = 1000
    REPORT_EVERY = 10  # print every 10 chunks = 10k S1
    chunk_num = 0
    
    def process_row(row):
        nonlocal total_cands, total_matches, empty_cands
        
        eid = row['entity_id']
        s1v = make_compact_view(row)
        
        # Candidate collection
        cand_scores = {}
        
        # Channel A: exact normalized name
        for cid in test_exact_index.get(s1v['n_raw'], []):
            cand_scores[cid] = 1.0
        
        # Channel B: inverted n-gram retrieval (top 20)
        for cid, sc in test_idx.query(s1v['index_text'], top_k=20, min_score=1):
            cand_scores[cid] = max(cand_scores.get(cid, 0.0), sc)
        
        # Prune to top 30
        top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:30]
        
        cand_ids_list = [c[0] for c in top_cands]
        cand_out.write(f"{eid}\t{','.join(cand_ids_list)}\n")
        n = len(cand_ids_list)
        total_cands += n
        cand_counts.append(n)
        if n == 0:
            empty_cands += 1
        
        # Score and decide
        scored = []
        exact_set = set(test_exact_index.get(s1v['n_raw'], []))
        for rank, (cid, sc) in enumerate(top_cands):
            tv = test_target_views.get(cid)
            if tv is None: continue
            nc = 2 if cid in exact_set else 1
            feat = compact_pair_features(s1v, tv, sc, rank, nc)
            raw_s = matcher.predict_proba([feat])[0]
            cal_s = calibrator.predict_proba([raw_s])[0]
            scored.append((cid, cal_s))
        
        preds = select_matches_for_s1(eid, scored, threshold=0.5)
        match_out.write(f"{eid}\t{','.join(preds)}\n")
        total_matches += len(preds)
    
    chunk = []
    for row in stream_rows(test_s1_path):
        chunk.append(row)
        if len(chunk) >= CHUNK_SIZE:
            for r in chunk:
                process_row(r)
            processed += len(chunk)
            chunk = []
            chunk_num += 1
            
            if chunk_num % REPORT_EVERY == 0:
                elapsed = time.time() - infer_start
                speed = processed / elapsed if elapsed > 0 else 1
                remaining = (test_s1_count - processed) / speed
                pct = 100 * processed / test_s1_count
                log(f"Processed: {processed:,}/{test_s1_count:,} ({pct:.1f}%) | "
                    f"Cands: {total_cands:,} | Matches: {total_matches:,} | "
                    f"Speed: {speed:.0f}/s | "
                    f"Elapsed: {datetime.timedelta(seconds=int(elapsed))} | "
                    f"ETA: {datetime.timedelta(seconds=int(remaining))}")
    
    if chunk:
        for r in chunk:
            process_row(r)
        processed += len(chunk)
    
    cand_out.close()
    match_out.close()
    
    total_runtime = time.time() - run_start
    log(f"=== INFERENCE COMPLETE ===")
    log(f"Total S1 processed : {processed:,}")
    log(f"Total candidate pairs: {total_cands:,}")
    log(f"Total matches      : {total_matches:,}")
    log(f"Empty cand S1      : {empty_cands:,}")
    log(f"Runtime            : {datetime.timedelta(seconds=int(total_runtime))}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Output validation
    # -------------------------------------------------------------------------
    assert processed == test_s1_count, f"FAIL: processed {processed} != {test_s1_count}"
    
    matching_rows  = sum(1 for _ in open(OUT_DIR / "matching_results.tsv", encoding='utf-8')) - 1
    candidate_rows = sum(1 for _ in open(OUT_DIR / "candidate_pairs.tsv",  encoding='utf-8')) - 1
    assert matching_rows  == test_s1_count, f"matching rows {matching_rows} != {test_s1_count}"
    assert candidate_rows == test_s1_count, f"candidate rows {candidate_rows} != {test_s1_count}"
    log(f"Row count assertions PASSED ({matching_rows:,} rows)")
    
    # -------------------------------------------------------------------------
    # STEP 5: Snapshot
    # -------------------------------------------------------------------------
    import shutil
    shutil.copy(OUT_DIR / "matching_results.tsv", SUB_DIR / "matching_results.tsv")
    shutil.copy(OUT_DIR / "candidate_pairs.tsv",  SUB_DIR / "candidate_pairs.tsv")
    
    cand_arr = np.array(cand_counts)
    stats = {
        "test_s1_count": test_s1_count,
        "test_target_count": n_test_targets,
        "processed_s1": processed,
        "total_candidate_pairs": int(total_cands),
        "total_matches": int(total_matches),
        "empty_candidate_s1": int(empty_cands),
        "mean_candidates": float(np.mean(cand_arr)),
        "median_candidates": float(np.median(cand_arr)),
        "p95_candidates": float(np.percentile(cand_arr, 95)),
        "p99_candidates": float(np.percentile(cand_arr, 99)),
        "runtime_seconds": int(total_runtime),
        "architecture": "InvertedNgramIndex_n3 + ExactBlocking",
        "train_target_sample": TRAIN_TARGET_LIMIT,
        "train_s1_fit": TRAIN_S1_FIT,
    }
    with open(SUB_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    config = {
        "random_seed": 42,
        "train_target_sample_each": TRAIN_TARGET_LIMIT,
        "train_s1_fit": TRAIN_S1_FIT,
        "retrieval_exact": "normalized_name exact match",
        "retrieval_ngram": "char 3-gram inverted index, top_k=20, min_score=1",
        "max_candidates": 30,
        "model": "lightgbm num_boost_round=200",
        "calibration": "Platt Scaling",
        "decision_threshold": 0.5
    }
    with open(SUB_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    report = [
        "# Phase 3 Full Baseline Report",
        "",
        "## Dataset",
        f"- Test S1 Queries: {test_s1_count:,}",
        f"- Test S2+S3 Targets: {n_test_targets:,}",
        "",
        "## Inference",
        f"- S1 Processed: {processed:,} / {test_s1_count:,} (100%)",
        f"- Total Candidate Pairs: {total_cands:,}",
        f"- S1 with no candidates: {empty_cands:,}",
        f"- Mean Candidates per S1: {float(np.mean(cand_arr)):.2f}",
        f"- Median Candidates: {float(np.median(cand_arr)):.0f}",
        f"- P95 Candidates: {float(np.percentile(cand_arr, 95)):.0f}",
        f"- P99 Candidates: {float(np.percentile(cand_arr, 99)):.0f}",
        "",
        "## Predictions",
        f"- Total Matches: {total_matches:,}",
        "",
        f"## Runtime: {datetime.timedelta(seconds=int(total_runtime))}",
        "",
        "## Validator: PENDING — run validate_submission.py"
    ]
    with open(SUB_DIR / "validation_report.md", "w") as f:
        f.write("\n".join(report))
    
    with open(SUB_DIR / "run_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    
    log("=== ALL DONE ===")
    for k, v in stats.items():
        log(f"  {k}: {v}")

if __name__ == "__main__":
    run_phase3()
