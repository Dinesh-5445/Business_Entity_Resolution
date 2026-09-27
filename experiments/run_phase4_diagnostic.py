"""
PHASE 4 DIAGNOSTIC SCRIPT
===========================
Runs a full leakage-safe validation on the training split to produce:
  - Candidate recall
  - Macro F0.5
  - Candidate count statistics
  - Singleton performance
  - False merge rate
  - S1→S2 / S1→S3 breakdown
  - False-positive pattern analysis
  - False-negative pattern analysis
  - Channel-level candidate recall breakdown

Uses an 80/20 stratified split of train_source1.tsv.
The 80% of S1 is used to train LightGBM.
The 20% of S1 is used for hold-out validation.
Targets (S2/S3) are NOT split — the model sees full S2/S3 during validation
(same as test time), which is leakage-safe because S2/S3 are the target corpus.

Output: docs/PHASE4_DIAGNOSTIC_REPORT.md
"""

import sys
import os
import json
import time
import datetime
import random
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'student_resource' / 'dataset'
DOCS_DIR = BASE_DIR / 'docs'
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_S1  = DATA_DIR / 'train' / 'train_source1.tsv'
TRAIN_S2  = DATA_DIR / 'train' / 'train_source2.tsv'
TRAIN_S3  = DATA_DIR / 'train' / 'train_source3.tsv'
TRAIN_GT  = DATA_DIR / 'train' / 'train_ground_truth.tsv'

FEATURES = [
    'num_channels', 'best_channel',
    's1_name_missing', 't_name_missing', 's1_addr_missing', 't_addr_missing',
    'name_jw', 'name_exact', 'core_exact',
    'addr_jw', 'num_exact', 'num_contra',
    'country_missing', 'country_agree', 'country_contra'
]


def ts():
    return datetime.datetime.now().strftime('%H:%M:%S')


def setup_conn(db_path=":memory:"):
    print(f"[{ts()}] Initializing DuckDB...")
    conn = duckdb.connect(db_path)
    conn.execute("PRAGMA memory_limit='10GB'")
    conn.execute("PRAGMA threads=8")
    conn.execute("CREATE MACRO alnum(x) AS regexp_replace(lower(x), '[^a-z0-9]', '', 'g')")
    conn.execute("CREATE MACRO norm(x) AS trim(regexp_replace(regexp_replace(lower(x), '[^a-z0-9 ]', ' ', 'g'), ' +', ' ', 'g'))")
    conn.execute("CREATE MACRO core(x) AS regexp_replace(norm(x), ' (inc|incorporated|corp|corporation|llc|ltd|limited|co|company|plc|gmbh|sa|nv|bv|srl|spa)$', '')")
    conn.execute("CREATE MACRO num_sig(x) AS regexp_replace(x, '[^0-9]', '', 'g')")
    return conn


def load_and_normalize(conn, name, path, source_tag=None, limit=None):
    lc = f" LIMIT {limit}" if limit else ""
    stag = f", '{source_tag}' as source" if source_tag else ""
    conn.execute(f"""
        CREATE TABLE raw_{name} AS 
        SELECT *{stag} FROM read_csv('{path}', sep='\t', header=True, ignore_errors=True){lc}
    """)
    conn.execute(f"""
        CREATE TABLE norm_{name} AS
        SELECT
            entity_id,
            coalesce(business_name, '') as raw_name,
            coalesce(business_address, '') as raw_addr,
            coalesce(lower(country), '') as country,
            alnum(business_name) as name_alnum,
            norm(business_name) as name_norm,
            core(business_name) as name_core,
            alnum(business_address) as addr_alnum,
            norm(business_address) as addr_norm,
            num_sig(business_address) as addr_num
        FROM raw_{name}
    """)
    conn.execute(f"CREATE INDEX idx_{name}_name_norm ON norm_{name}(name_norm)")
    conn.execute(f"CREATE INDEX idx_{name}_name_core ON norm_{name}(name_core)")
    conn.execute(f"CREATE INDEX idx_{name}_addr_norm ON norm_{name}(addr_norm)")
    print(f"[{ts()}] Loaded and indexed norm_{name}: {conn.execute(f'SELECT COUNT(*) FROM norm_{name}').fetchone()[0]:,} rows")


def generate_candidates(conn, s1_name, tgt_name, out_name):
    s1 = f"norm_{s1_name}"
    t  = f"norm_{tgt_name}"
    conn.execute(f"""
        CREATE TABLE {out_name} AS
        SELECT s1_id, t_id, min(channel) as best_channel, count(*) as num_channels
        FROM (
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 1 as channel
            FROM {s1} s1 JOIN {t} t ON s1.name_norm = t.name_norm WHERE s1.name_norm != ''
            UNION ALL
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 2 as channel
            FROM {s1} s1 JOIN {t} t ON s1.name_core = t.name_core WHERE s1.name_core != ''
            UNION ALL
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 3 as channel
            FROM {s1} s1 JOIN {t} t ON s1.addr_norm = t.addr_norm WHERE s1.addr_norm != ''
            UNION ALL
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 4 as channel
            FROM {s1} s1 JOIN {t} t ON s1.name_alnum = t.name_alnum AND s1.addr_num = t.addr_num
            WHERE s1.name_alnum != '' AND s1.addr_num != ''
            UNION ALL
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 5 as channel
            FROM {s1} s1 JOIN {t} t ON s1.country = t.country AND s1.name_core = t.name_core
            WHERE s1.country != '' AND s1.name_core != ''
        )
        GROUP BY s1_id, t_id
    """)
    n = conn.execute(f"SELECT COUNT(*) FROM {out_name}").fetchone()[0]
    print(f"[{ts()}] Generated {n:,} candidates in {out_name}")
    return n


def extract_features(conn, cand_name, s1_name, tgt_name, out_name):
    s1 = f"norm_{s1_name}"
    t  = f"norm_{tgt_name}"
    conn.execute(f"""
        CREATE TABLE {out_name} AS
        SELECT
            c.s1_id, c.t_id, c.num_channels, c.best_channel,
            CASE WHEN s1.raw_name = '' THEN 1 ELSE 0 END as s1_name_missing,
            CASE WHEN t.raw_name  = '' THEN 1 ELSE 0 END as t_name_missing,
            CASE WHEN s1.raw_addr = '' THEN 1 ELSE 0 END as s1_addr_missing,
            CASE WHEN t.raw_addr  = '' THEN 1 ELSE 0 END as t_addr_missing,
            jaro_winkler_similarity(s1.name_alnum, t.name_alnum) as name_jw,
            CASE WHEN s1.name_norm = t.name_norm AND s1.name_norm != '' THEN 1.0 ELSE 0.0 END as name_exact,
            CASE WHEN s1.name_core = t.name_core AND s1.name_core != '' THEN 1.0 ELSE 0.0 END as core_exact,
            jaro_winkler_similarity(s1.addr_alnum, t.addr_alnum) as addr_jw,
            CASE WHEN s1.addr_num = t.addr_num AND s1.addr_num != '' THEN 1.0 ELSE 0.0 END as num_exact,
            CASE WHEN s1.addr_num != t.addr_num AND s1.addr_num != '' AND t.addr_num != '' THEN 1.0 ELSE 0.0 END as num_contra,
            CASE WHEN s1.country = '' OR t.country = ''  THEN 1.0 ELSE 0.0 END as country_missing,
            CASE WHEN s1.country = t.country AND s1.country != '' THEN 1.0 ELSE 0.0 END as country_agree,
            CASE WHEN s1.country != t.country AND s1.country != '' AND t.country != '' THEN 1.0 ELSE 0.0 END as country_contra
        FROM {cand_name} c
        JOIN {s1} s1 ON c.s1_id = s1.entity_id
        JOIN {t}  t  ON c.t_id  = t.entity_id
    """)
    print(f"[{ts()}] Extracted features for {out_name}")


def load_gt(conn, gt_path):
    conn.execute("CREATE TABLE gt_raw (source1_entity_id VARCHAR, matched_entity_ids VARCHAR)")
    conn.execute(f"COPY gt_raw FROM '{gt_path}' (DELIMITER '\t', HEADER TRUE)")
    conn.execute("""
        CREATE TABLE gt_pairs AS
        SELECT source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as t_id
        FROM gt_raw WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    """)
    n = conn.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
    print(f"[{ts()}] Ground truth: {n:,} positive pairs")
    return n


def compute_candidate_recall(conn, cand_name, val_s1_ids, gt_pairs_df):
    """Compute what % of true positive pairs appear in candidate set."""
    val_gt = gt_pairs_df[gt_pairs_df['s1_id'].isin(val_s1_ids)]
    if len(val_gt) == 0:
        return 0.0, 0, 0

    cand_df = conn.execute(f"SELECT s1_id, t_id FROM {cand_name}").df()
    cand_set = set(zip(cand_df['s1_id'], cand_df['t_id']))
    
    found = sum(1 for _, r in val_gt.iterrows() if (r['s1_id'], r['t_id']) in cand_set)
    total = len(val_gt)
    return found / total, found, total


def channel_recall_breakdown(conn, cand_name, val_s1_ids, gt_pairs_df):
    """Per-channel candidate recall."""
    val_gt = gt_pairs_df[gt_pairs_df['s1_id'].isin(val_s1_ids)]
    cand_df = conn.execute(f"SELECT s1_id, t_id, best_channel, num_channels FROM {cand_name}").df()
    cand_map = {}
    for _, r in cand_df.iterrows():
        cand_map[(r['s1_id'], r['t_id'])] = r['best_channel']

    found_by_channel = {1:0, 2:0, 3:0, 4:0, 5:0}
    not_found = 0
    for _, r in val_gt.iterrows():
        key = (r['s1_id'], r['t_id'])
        if key in cand_map:
            ch = int(cand_map[key])
            found_by_channel[ch] = found_by_channel.get(ch, 0) + 1
        else:
            not_found += 1
    return found_by_channel, not_found


def f_beta_score(precision, recall, beta=0.5):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_macro_f05(val_s1_ids, gt_df, pred_df):
    """
    Compute macro F0.5 per S1 entity.
    pred_df: DataFrame with columns [s1_id, t_id]
    gt_df: DataFrame with columns [s1_id, t_id]
    """
    gt_by_s1 = gt_df.groupby('s1_id')['t_id'].apply(set).to_dict()
    pred_by_s1 = pred_df.groupby('s1_id')['t_id'].apply(set).to_dict() if len(pred_df) > 0 else {}

    f05_scores = []
    for s1 in val_s1_ids:
        truth = gt_by_s1.get(s1, set())
        preds = pred_by_s1.get(s1, set())

        tp = len(truth & preds)
        fp = len(preds - truth)
        fn = len(truth - preds)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05  = f_beta_score(prec, rec, beta=0.5)
        f05_scores.append(f05)

    return np.mean(f05_scores), f05_scores


def analyze_false_positives(pred_df, gt_df, conn, s1_name, tgt_name, top_n=200):
    """Return top false-positive pairs with their raw fields."""
    merged = pred_df.merge(gt_df.assign(is_tp=True), on=['s1_id','t_id'], how='left')
    fps = merged[merged['is_tp'].isna()].sort_values('score', ascending=False).head(top_n)

    # Fetch raw fields for inspection
    s1_fields = conn.execute(f"SELECT entity_id, raw_name, raw_addr, country FROM norm_{s1_name}").df()
    t_fields  = conn.execute(f"SELECT entity_id, raw_name, raw_addr, country FROM norm_{tgt_name}").df()
    s1_fields.columns = ['s1_id','s1_name','s1_addr','s1_country']
    t_fields.columns  = ['t_id', 't_name', 't_addr', 't_country']

    fps = fps.merge(s1_fields, on='s1_id', how='left').merge(t_fields, on='t_id', how='left')
    return fps


def analyze_false_negatives(val_gt, cand_df, pred_df, conn, s1_name, tgt_name):
    """Find true positives that were in candidates but rejected by LightGBM."""
    cand_set = set(zip(cand_df['s1_id'], cand_df['t_id']))
    pred_set = set(zip(pred_df['s1_id'], pred_df['t_id']))

    fns_in_cands = val_gt[
        val_gt.apply(lambda r: (r['s1_id'], r['t_id']) in cand_set, axis=1) &
        ~val_gt.apply(lambda r: (r['s1_id'], r['t_id']) in pred_set, axis=1)
    ].head(200)

    s1_fields = conn.execute(f"SELECT entity_id, raw_name, raw_addr, country FROM norm_{s1_name}").df()
    t_fields  = conn.execute(f"SELECT entity_id, raw_name, raw_addr, country FROM norm_{tgt_name}").df()
    s1_fields.columns = ['s1_id','s1_name','s1_addr','s1_country']
    t_fields.columns  = ['t_id', 't_name', 't_addr', 't_country']

    fns_in_cands = fns_in_cands.merge(s1_fields, on='s1_id', how='left').merge(t_fields, on='t_id', how='left')
    return fns_in_cands


def run_diagnostic():
    run_start = time.time()
    db_path = str(BASE_DIR / 'experiments' / 'phase4_diag.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = setup_conn(db_path)

    # ------------------------------------------------------------------
    # 1. LOAD ALL DATA
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === LOADING DATA ===")
    load_and_normalize(conn, 's2', TRAIN_S2, source_tag='S2')
    load_and_normalize(conn, 's3', TRAIN_S3, source_tag='S3')
    conn.execute("""
        CREATE TABLE norm_targets AS
        SELECT *, 'S2' as source FROM norm_s2
        UNION ALL
        SELECT *, 'S3' as source FROM norm_s3
    """)
    print(f"[{ts()}] Total targets: {conn.execute('SELECT COUNT(*) FROM norm_targets').fetchone()[0]:,}")

    # Load full S1 and GT to do split
    s1_df = pd.read_csv(TRAIN_S1, sep='\t')
    gt_df_raw = pd.read_csv(TRAIN_GT, sep='\t')
    gt_df_raw.columns = ['s1_id', 'matched_entity_ids']

    # Build GT as pairs
    gt_pairs = []
    for _, r in gt_df_raw.iterrows():
        if pd.notna(r['matched_entity_ids']) and r['matched_entity_ids'] != '':
            for tid in str(r['matched_entity_ids']).split(','):
                gt_pairs.append({'s1_id': r['s1_id'], 't_id': tid.strip()})
    gt_pairs_df = pd.DataFrame(gt_pairs)
    print(f"[{ts()}] Total GT positive pairs: {len(gt_pairs_df):,}")

    # 80/20 stratified split on whether entity has matches
    s1_ids_all = s1_df['entity_id'].tolist()
    s1_with_gt = set(gt_df_raw[gt_df_raw['matched_entity_ids'].notna() & (gt_df_raw['matched_entity_ids'] != '')]['s1_id'])
    
    has_match   = [e for e in s1_ids_all if e in s1_with_gt]
    no_match    = [e for e in s1_ids_all if e not in s1_with_gt]

    random.seed(42)
    random.shuffle(has_match)
    random.shuffle(no_match)

    split_hm = int(len(has_match) * 0.8)
    split_nm = int(len(no_match)  * 0.8)
    
    train_s1_ids = set(has_match[:split_hm] + no_match[:split_nm])
    val_s1_ids   = set(has_match[split_hm:] + no_match[split_nm:])
    
    print(f"[{ts()}] Train S1: {len(train_s1_ids):,} | Val S1: {len(val_s1_ids):,}")

    # Write split TSVs to temp files
    tmp_dir = BASE_DIR / 'experiments' / 'diag_tmp'
    tmp_dir.mkdir(exist_ok=True)
    
    train_s1_df = s1_df[s1_df['entity_id'].isin(train_s1_ids)]
    val_s1_df   = s1_df[s1_df['entity_id'].isin(val_s1_ids)]
    train_s1_df.to_csv(tmp_dir / 'train_s1.tsv', sep='\t', index=False)
    val_s1_df.to_csv(tmp_dir / 'val_s1.tsv',   sep='\t', index=False)

    # ------------------------------------------------------------------
    # 2. TRAIN SPLIT CANDIDATE GENERATION
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === TRAINING CANDIDATE GENERATION ===")
    load_and_normalize(conn, 'tr_s1', tmp_dir / 'train_s1.tsv')
    generate_candidates(conn, 'tr_s1', 'targets', 'tr_cands')
    extract_features(conn, 'tr_cands', 'tr_s1', 'targets', 'tr_feats')

    # Label training pairs entirely in DuckDB (avoids pulling 49M rows to Python)
    print(f"[{ts()}] Registering GT pairs in DuckDB...")
    conn.execute("CREATE TABLE gt_pairs_tbl AS SELECT s1_id, t_id FROM gt_pairs_df")
    
    # Join features with GT labels in SQL
    conn.execute("""
        CREATE TABLE tr_labeled AS
        SELECT f.*, CASE WHEN g.t_id IS NOT NULL THEN 1 ELSE 0 END as label
        FROM tr_feats f
        LEFT JOIN gt_pairs_tbl g ON f.s1_id = g.s1_id AND f.t_id = g.t_id
    """)
    
    n_pos = conn.execute("SELECT COUNT(*) FROM tr_labeled WHERE label=1").fetchone()[0]
    n_neg = conn.execute("SELECT COUNT(*) FROM tr_labeled WHERE label=0").fetchone()[0]
    print(f"[{ts()}] Training pairs: {n_pos+n_neg:,} | Positives: {n_pos:,} | Negatives: {n_neg:,}")
    
    # Sample: all positives + up to 5M random negatives (keeps memory manageable)
    NEG_SAMPLE = 5_000_000
    print(f"[{ts()}] Sampling {NEG_SAMPLE:,} negatives + all {n_pos:,} positives for training...")
    conn.execute(f"""
        CREATE TABLE tr_sample AS
        SELECT * FROM tr_labeled WHERE label=1
        UNION ALL
        SELECT * FROM tr_labeled WHERE label=0 USING SAMPLE {NEG_SAMPLE}
    """)
    
    tr_feats_df = conn.execute("SELECT * FROM tr_sample").df()
    print(f"[{ts()}] Training set: {len(tr_feats_df):,} pairs | Positives: {tr_feats_df['label'].sum():,}")

    # ------------------------------------------------------------------
    # 3. TRAIN LIGHTGBM
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === TRAINING LIGHTGBM ===")
    X_tr = tr_feats_df[FEATURES].fillna(0)
    y_tr = tr_feats_df['label']

    model = LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)
    print(f"[{ts()}] Model trained.")

    # ------------------------------------------------------------------
    # 4. VALIDATION SPLIT CANDIDATE GENERATION
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === VALIDATION CANDIDATE GENERATION ===")
    load_and_normalize(conn, 'val_s1', tmp_dir / 'val_s1.tsv')
    generate_candidates(conn, 'val_s1', 'targets', 'val_cands')
    extract_features(conn, 'val_cands', 'val_s1', 'targets', 'val_feats')

    # ------------------------------------------------------------------
    # 5. CANDIDATE RECALL (computed in DuckDB to avoid huge pandas sets)
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === CANDIDATE RECALL ANALYSIS ===")
    val_gt = gt_pairs_df[gt_pairs_df['s1_id'].isin(val_s1_ids)].copy()
    
    # Register val GT in DuckDB
    conn.execute("CREATE TABLE val_gt_tbl AS SELECT s1_id, t_id FROM val_gt")
    
    # Count how many GT pairs appear in val_cands
    n_found = conn.execute("""
        SELECT COUNT(*) FROM val_gt_tbl g
        JOIN val_cands c ON g.s1_id = c.s1_id AND g.t_id = c.t_id
    """).fetchone()[0]
    n_total_gt = len(val_gt)
    n_missed   = n_total_gt - n_found
    cand_recall = n_found / n_total_gt if n_total_gt > 0 else 0.0
    print(f"Candidate recall: {cand_recall:.4f} ({n_found}/{n_total_gt} | {n_missed} missed)")

    # Channel breakdown via SQL
    channel_names = {1:"Exact name_norm", 2:"Exact core-name", 3:"Exact addr_norm", 4:"name_alnum+addr_num", 5:"country+core-name"}
    ch_df = conn.execute("""
        SELECT c.best_channel, COUNT(*) as cnt
        FROM val_gt_tbl g
        JOIN val_cands c ON g.s1_id = c.s1_id AND g.t_id = c.t_id
        GROUP BY c.best_channel
    """).df()
    found_by_ch = dict(zip(ch_df['best_channel'].astype(int), ch_df['cnt']))
    for ch in [1,2,3,4,5]:
        found_by_ch.setdefault(ch, 0)

    # Sample missed pairs for analysis
    missed_df = conn.execute("""
        SELECT g.s1_id, g.t_id FROM val_gt_tbl g
        LEFT JOIN val_cands c ON g.s1_id = c.s1_id AND g.t_id = c.t_id
        WHERE c.t_id IS NULL
        LIMIT 500
    """).df()

    # Fetch val cand stats (counts only, not all pairs)
    val_cand_count_df = conn.execute("""
        SELECT s1_id, COUNT(*) as cnt FROM val_cands GROUP BY s1_id
    """).df()
    
    # Fetch val features — only a sample for FP/FN analysis, score all via chunking
    val_feats_df = conn.execute(f"SELECT * FROM val_feats LIMIT 5000000").df()

    # ------------------------------------------------------------------
    # 6. PREDICT ON VALIDATION SET (chunked to avoid OOM)
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === VALIDATION INFERENCE (chunked) ===")
    THRESHOLD = 0.5
    CHUNK_SIZE = 2_000_000
    val_n_total = conn.execute("SELECT COUNT(*) FROM val_feats").fetchone()[0]
    offset = 0
    all_val_preds = []
    while offset < val_n_total:
        chunk = conn.execute(
            f"SELECT * FROM val_feats LIMIT {CHUNK_SIZE} OFFSET {offset}"
        ).df()
        X_chunk = chunk[FEATURES].fillna(0)
        chunk['score'] = model.predict_proba(X_chunk)[:, 1]
        preds_chunk = chunk[chunk['score'] > THRESHOLD][['s1_id','t_id','score']]
        all_val_preds.append(preds_chunk)
        offset += CHUNK_SIZE
        print(f"  Scored {min(offset, val_n_total):,}/{val_n_total:,} val pairs...")

    val_preds = pd.concat(all_val_preds) if all_val_preds else pd.DataFrame(columns=['s1_id','t_id','score'])
    print(f"Predictions above threshold {THRESHOLD}: {len(val_preds):,}")

    # ------------------------------------------------------------------
    # 7. COMPUTE MACRO F0.5
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === COMPUTING MACRO F0.5 ===")
    macro_f05, f05_per_s1 = compute_macro_f05(list(val_s1_ids), val_gt, val_preds)
    print(f"Macro F0.5: {macro_f05:.4f}")

    # ------------------------------------------------------------------
    # 8. CANDIDATE COUNT STATISTICS (from pre-aggregated DuckDB query)
    # ------------------------------------------------------------------
    cand_counts = val_cand_count_df.set_index('s1_id')['cnt'].reindex(list(val_s1_ids), fill_value=0)
    mean_cands   = cand_counts.mean()
    median_cands = cand_counts.median()
    p95_cands    = np.percentile(cand_counts.values, 95)
    p99_cands    = np.percentile(cand_counts.values, 99)
    empty_cands  = (cand_counts == 0).sum()
    print(f"Candidates/S1: mean={mean_cands:.1f} | median={median_cands:.0f} | p95={p95_cands:.0f} | p99={p99_cands:.0f}")
    print(f"S1 with 0 candidates: {empty_cands:,} / {len(val_s1_ids):,}")

    # ------------------------------------------------------------------
    # 9. SINGLETON & FALSE MERGE
    # ------------------------------------------------------------------
    true_singletons = set(val_s1_ids) - set(val_gt['s1_id'].unique())
    pred_s1_set = set(val_preds['s1_id'].unique())

    singleton_fp = len(true_singletons & pred_s1_set)
    singleton_total = len(true_singletons)
    singleton_prec = 1 - (singleton_fp / singleton_total) if singleton_total > 0 else 1.0

    # False merge rate (S1 predicted ≥2 matches where true answer is exactly 0 or 1)
    pred_match_counts = val_preds.groupby('s1_id').size()
    false_merges = 0
    for s1, count in pred_match_counts.items():
        if count > 1:
            true_count = len(val_gt[val_gt['s1_id']==s1])
            if true_count <= 1:
                false_merges += 1

    print(f"Singleton false-positive rate: {1-singleton_prec:.4f} ({singleton_fp}/{singleton_total})")
    print(f"False merge count: {false_merges}")

    # ------------------------------------------------------------------
    # 10. S1→S2 / S1→S3 BREAKDOWN
    # ------------------------------------------------------------------
    # Derive target source from prefix (S2- or S3-)
    val_preds_enriched = val_preds.copy()
    val_preds_enriched['tgt_source'] = val_preds_enriched['t_id'].str[:2]
    val_gt_enriched = val_gt.copy()
    val_gt_enriched['tgt_source'] = val_gt_enriched['t_id'].str[:2]

    src_s2_macro = src_s3_macro = 0.0
    for src in ['S2', 'S3']:
        src_pred = val_preds_enriched[val_preds_enriched['tgt_source']==src]
        src_gt   = val_gt_enriched[val_gt_enriched['tgt_source']==src]
        src_macro, _ = compute_macro_f05(list(val_s1_ids), src_gt, src_pred)
        if src == 'S2': src_s2_macro = src_macro
        if src == 'S3': src_s3_macro = src_macro
        print(f"S1→{src} Macro F0.5: {src_macro:.4f} | GT pairs: {len(src_gt):,} | Predicted: {len(src_pred):,}")

    # ------------------------------------------------------------------
    # 11. FALSE POSITIVE ANALYSIS
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === FALSE POSITIVE ANALYSIS ===")
    gt_set = set(zip(val_gt['s1_id'], val_gt['t_id']))
    val_preds_enriched['is_tp'] = val_preds_enriched.apply(
        lambda r: (r['s1_id'], r['t_id']) in gt_set, axis=1
    )
    fps = val_preds_enriched[~val_preds_enriched['is_tp']].sort_values('score', ascending=False)
    
    # Fetch raw fields (only top 500 FP/FN targets to avoid memory issues)
    top_fp_tids = fps.head(500)['t_id'].tolist()
    top_fp_s1ids = fps.head(500)['s1_id'].tolist()
    s1_raw = conn.execute(
        f"SELECT entity_id, raw_name, raw_addr, country FROM norm_val_s1 WHERE entity_id IN ({','.join(repr(x) for x in top_fp_s1ids)})"
    ).df() if top_fp_s1ids else pd.DataFrame(columns=['entity_id','raw_name','raw_addr','country'])
    t_raw  = conn.execute(
        f"SELECT entity_id, raw_name, raw_addr, country FROM norm_targets WHERE entity_id IN ({','.join(repr(x) for x in top_fp_tids)})"
    ).df() if top_fp_tids else pd.DataFrame(columns=['entity_id','raw_name','raw_addr','country'])
    s1_raw.columns = ['s1_id','s1_name','s1_addr','s1_country']
    t_raw.columns  = ['t_id','t_name','t_addr','t_country']

    fps_enriched = fps.head(200).merge(s1_raw, on='s1_id', how='left').merge(t_raw, on='t_id', how='left')

    # Pattern classification
    def classify_fp(row):
        try:
            s1n = str(row.get('s1_name','')).lower()
            tn  = str(row.get('t_name','')).lower()
            s1a = str(row.get('s1_addr','')).lower()
            ta  = str(row.get('t_addr','')).lower()
            if s1n == tn or s1n[:20] == tn[:20]:
                if s1a and ta and s1a[:15] != ta[:15]:
                    return "same_name_diff_addr"
                return "identical_name_diff_entity"
            if s1n.split() and tn.split() and s1n.split()[0] == tn.split()[0]:
                return "same_first_token_diff_entity"
            if len(s1n) < 8 or len(tn) < 8:
                return "short_generic_name"
            if s1a and ta and (s1a[:15] == ta[:15]):
                return "same_addr_diff_name"
            return "other"
        except:
            return "other"

    fps_enriched['fp_pattern'] = fps_enriched.apply(classify_fp, axis=1)
    fp_summary = fps_enriched['fp_pattern'].value_counts()
    print(f"Top FP patterns (top 200 FPs):\n{fp_summary.to_string()}")

    # ------------------------------------------------------------------
    # 12. FALSE NEGATIVE ANALYSIS (In-Candidate) — via DuckDB
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === FALSE NEGATIVE ANALYSIS (in-candidates) ===")
    conn.execute("CREATE TABLE val_preds_tbl AS SELECT s1_id, t_id FROM val_preds")
    
    fn_count = conn.execute("""
        SELECT COUNT(*) FROM val_gt_tbl g
        JOIN val_cands c ON g.s1_id=c.s1_id AND g.t_id=c.t_id
        LEFT JOIN val_preds_tbl p ON g.s1_id=p.s1_id AND g.t_id=p.t_id
        WHERE p.t_id IS NULL
    """).fetchone()[0]
    
    fns_in_cands = conn.execute("""
        SELECT g.s1_id, g.t_id FROM val_gt_tbl g
        JOIN val_cands c ON g.s1_id=c.s1_id AND g.t_id=c.t_id
        LEFT JOIN val_preds_tbl p ON g.s1_id=p.s1_id AND g.t_id=p.t_id
        WHERE p.t_id IS NULL
        LIMIT 200
    """).df()
    
    # Join with features from the first sampled chunk
    fns_enriched = fns_in_cands.merge(
        val_feats_df[['s1_id','t_id','score','name_jw','addr_jw','name_exact','core_exact','num_exact','num_contra','country_agree']],
        on=['s1_id','t_id'], how='left'
    )
    print(f"FN in candidates: {fn_count:,}")
    if len(fns_enriched) > 0 and 'score' in fns_enriched.columns:
        print(f"  Avg LGB score for FN-in-cands (sample): {fns_enriched['score'].mean():.4f}")
        print(f"  Avg name_jw: {fns_enriched['name_jw'].mean():.4f}")
        print(f"  Avg addr_jw: {fns_enriched['addr_jw'].mean():.4f}")

    # ------------------------------------------------------------------
    # 13. MISSED GT PAIRS (NOT IN CANDIDATES AT ALL)
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === MISSED PAIRS (not in candidates) ===")
    top_miss_tids = missed_df['t_id'].tolist() if len(missed_df)>0 else []
    top_miss_s1ids = missed_df['s1_id'].tolist() if len(missed_df)>0 else []
    if top_miss_s1ids:
        s1_miss_raw = conn.execute(
            f"SELECT entity_id, raw_name, raw_addr, country FROM norm_val_s1 WHERE entity_id IN ({','.join(repr(x) for x in top_miss_s1ids[:200])})"
        ).df()
        t_miss_raw  = conn.execute(
            f"SELECT entity_id, raw_name, raw_addr, country FROM norm_targets WHERE entity_id IN ({','.join(repr(x) for x in top_miss_tids[:200])})"
        ).df()
        s1_miss_raw.columns = ['s1_id','s1_name','s1_addr','s1_country']
        t_miss_raw.columns  = ['t_id','t_name','t_addr','t_country']
        missed_enriched = missed_df.head(200).merge(s1_miss_raw, on='s1_id', how='left').merge(t_miss_raw, on='t_id', how='left')
        print(f"Sample of missed pairs ({len(missed_df)} total):")
        for _, r in missed_enriched.head(10).iterrows():
            print(f"  S1: '{r.get('s1_name','')}' / '{r.get('s1_addr','')}' → T: '{r.get('t_name','')}' / '{r.get('t_addr','')}' (ctry: {r.get('s1_country','')}→{r.get('t_country','')})")
    else:
        missed_enriched = pd.DataFrame()
        print("All GT pairs found in candidates!")

    # ------------------------------------------------------------------
    # 14. WRITE REPORT
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === WRITING DIAGNOSTIC REPORT ===")
    
    src_s2_macro = 0.0
    src_s3_macro = 0.0
    for src in ['S2', 'S3']:
        src_pred = val_preds_enriched[val_preds_enriched['tgt_source']==src]
        src_gt   = val_gt_enriched[val_gt_enriched['tgt_source']==src]
        m, _ = compute_macro_f05(list(val_s1_ids), src_gt, src_pred)
        if src == 'S2': src_s2_macro = m
        if src == 'S3': src_s3_macro = m

    # Determine dominant bottleneck
    if cand_recall < 0.7:
        bottleneck = "A — CANDIDATE GENERATION (recall too low; true matches not reaching LightGBM)"
        first_exp  = "Improve candidate channels: add token-level blocking, char n-gram channel, address-specific channels"
    elif cand_recall >= 0.7 and macro_f05 < 0.55 and len(fns_in_cands) > len(fps):
        bottleneck = "B — MATCHER (candidates present, LightGBM rejecting true matches)"
        first_exp  = "Hard-negative mining + richer pair features (token Jaccard, TF-IDF cosine)"
    elif singleton_fp / max(1, singleton_total) > 0.10:
        bottleneck = "E — SINGLETON HANDLING (too many false merges on non-matching S1s)"
        first_exp  = "Singleton gate + threshold sweep; calibration improvement"
    else:
        bottleneck = "D — FALSE MERGES / CALIBRATION (too many FPs being accepted)"
        first_exp  = "Threshold sweep; hard-negative mining with same-name-diff-addr examples"

    runtime = time.time() - run_start

    report = f"""# PHASE 4 DIAGNOSTIC REPORT
Generated: {datetime.datetime.now().isoformat()}
Runtime: {runtime:.0f}s

---

## A. Phase 3 Configuration

| Property | Value |
|---|---|
| Architecture | DuckDB Native Vectorized Pipeline |
| Candidate Channels | 5 (exact name_norm, exact core-name, exact addr_norm, name_alnum+addr_num, country+core-name) |
| LightGBM n_estimators | 150 |
| Decision threshold | 0.5 |
| Training sample | 500k rows from S2/S3 (LIMIT 500k on S2 and S3 each) |
| Features | {len(FEATURES)} features |

## B. Phase 3 Leaderboard Baseline
- **Leaderboard Macro F0.5: 0.505**

## C. Local Validation Metrics (80/20 train split, leakage-safe)

| Metric | Value |
|---|---|
| Macro F0.5 | {macro_f05:.4f} |
| Candidate Recall | {cand_recall:.4f} ({n_found}/{n_total_gt}) |
| Mean Candidates/S1 | {mean_cands:.1f} |
| Median Candidates/S1 | {median_cands:.0f} |
| P95 Candidates/S1 | {p95_cands:.0f} |
| P99 Candidates/S1 | {p99_cands:.0f} |
| S1 with 0 candidates | {empty_cands:,} / {len(val_s1_ids):,} ({empty_cands/len(val_s1_ids)*100:.1f}%) |

## D. Candidate Count Distribution

Mean: {mean_cands:.1f} | Median: {median_cands:.0f} | P95: {p95_cands:.0f} | P99: {p99_cands:.0f}

## E. Singleton Performance

| Metric | Value |
|---|---|
| True singletons in val | {singleton_total:,} |
| Singletons with FP predictions | {singleton_fp:,} |
| Singleton false-positive rate | {singleton_fp/max(1,singleton_total)*100:.2f}% |

## F. False Merge Rate

| Metric | Value |
|---|---|
| S1 predicted with ≥2 matches but true count ≤1 | {false_merges:,} |

## G. S1→S2 Performance

| Metric | Value |
|---|---|
| S1→S2 Macro F0.5 | {src_s2_macro:.4f} |

## H. S1→S3 Performance

| Metric | Value |
|---|---|
| S1→S3 Macro F0.5 | {src_s3_macro:.4f} |

## I. Candidate-Generation Failure Analysis

| Channel | True Pairs Retrieved | % of Found |
|---|---|---|
| Ch1: Exact name_norm | {found_by_ch.get(1,0)} | {found_by_ch.get(1,0)/max(1,n_found)*100:.1f}% |
| Ch2: Exact core-name | {found_by_ch.get(2,0)} | {found_by_ch.get(2,0)/max(1,n_found)*100:.1f}% |
| Ch3: Exact addr_norm | {found_by_ch.get(3,0)} | {found_by_ch.get(3,0)/max(1,n_found)*100:.1f}% |
| Ch4: name_alnum+addr_num | {found_by_ch.get(4,0)} | {found_by_ch.get(4,0)/max(1,n_found)*100:.1f}% |
| Ch5: country+core-name | {found_by_ch.get(5,0)} | {found_by_ch.get(5,0)/max(1,n_found)*100:.1f}% |
| **NOT FOUND (missed)** | **{n_missed}** | **{n_missed/max(1,n_total_gt)*100:.1f}% of all GT** |

## J. Matcher Failure Analysis

| Metric | Value |
|---|---|
| GT pairs present in candidates | {n_found:,} |
| GT pairs predicted correctly (TP) | — |
| GT pairs in candidates but rejected (FN-in-cands) | {len(fns_in_cands):,} |
| Avg LGB score for FN-in-cands | {fns_enriched['score'].mean():.4f if len(fns_enriched)>0 else 'N/A'} |
| Avg name_jw for FN-in-cands | {fns_enriched['name_jw'].mean():.4f if len(fns_enriched)>0 else 'N/A'} |
| Avg addr_jw for FN-in-cands | {fns_enriched['addr_jw'].mean():.4f if len(fns_enriched)>0 else 'N/A'} |

## K. Decision/Calibration Analysis

- Threshold used: {THRESHOLD}
- Total predictions: {len(val_preds):,}
- True positive predictions: approx {int(val_preds_enriched['is_tp'].sum()):,}
- False positive predictions: {int(len(fps)):,}

## L. Top False-Positive Patterns (top 200 FPs)

{fp_summary.to_string()}

## M. Sample Missed Positive Pairs (Not in Candidates)

{chr(10).join([f"- S1: '{r.get('s1_name','')}' / '{r.get('s1_addr','')}' → T: '{r.get('t_name','')}' / '{r.get('t_addr','')}'  (S1_ctry={r.get('s1_country','')}, T_ctry={r.get('t_country','')})" for _, r in missed_enriched.head(20).iterrows()] if len(missed_df) > 0 else ["All GT pairs found in candidates!"])}

## N. Dominant Bottleneck

**{bottleneck}**

- Candidate Recall: {cand_recall:.4f}
- FN-in-cands: {len(fns_in_cands):,} (matcher failed on candidates present)
- FP count: {len(fps):,}
- Singleton FP rate: {singleton_fp/max(1,singleton_total)*100:.2f}%

## O. Recommended First Phase 4 Experiment

**{first_exp}**

---
*Report generated by experiments/run_phase4_diagnostic.py*
"""

    report_path = DOCS_DIR / 'PHASE4_DIAGNOSTIC_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[{ts()}] Report written to {report_path}")
    print(f"\n[{ts()}] === DIAGNOSTIC COMPLETE ===")
    print(f"Macro F0.5:        {macro_f05:.4f}")
    print(f"Candidate Recall:  {cand_recall:.4f}")
    print(f"Dominant Issue:    {bottleneck}")

    # Write FP and FN detail files
    fps_path = DOCS_DIR / 'PHASE4_FALSE_POSITIVE_ANALYSIS.md'
    with open(fps_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4 False Positive Analysis\n\n")
        f.write(fps_enriched.head(50)[['s1_id','t_id','score','s1_name','t_name','s1_addr','t_addr','s1_country','t_country','fp_pattern']].to_markdown(index=False))
    
    fns_path = DOCS_DIR / 'PHASE4_FALSE_NEGATIVE_ANALYSIS.md'
    with open(fns_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4 False Negative Analysis (in-candidates, rejected by matcher)\n\n")
        if len(fns_enriched) > 0:
            f.write(fns_enriched.head(50)[['s1_id','t_id','score','s1_name','t_name','s1_addr','t_addr','s1_country','t_country']].to_markdown(index=False))

    print(f"[{ts()}] FP analysis: {fps_path}")
    print(f"[{ts()}] FN analysis: {fns_path}")


if __name__ == '__main__':
    run_diagnostic()
