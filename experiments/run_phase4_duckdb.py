import sys
import os
import json
import time
import datetime
import shutil
from pathlib import Path
import pandas as pd
import duckdb
from lightgbm import LGBMClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'student_resource' / 'dataset'
SUB_DIR  = BASE_DIR / 'submissions' / 'phase_4_candidate_recall'
OUT_DIR  = BASE_DIR / 'output'

SUB_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def setup_duckdb(db_path=":memory:"):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Initializing DuckDB at {db_path}...")
    conn = duckdb.connect(db_path)
    
    conn.execute("PRAGMA memory_limit='12GB'")
    conn.execute("PRAGMA threads=8")
    
    # 1. Alphanumeric compact
    conn.execute("CREATE MACRO alnum(x) AS regexp_replace(lower(x), '[^a-z0-9]', '', 'g')")
    
    # 2. Normalized text (punctuation to space, compact spaces)
    conn.execute("CREATE MACRO norm(x) AS trim(regexp_replace(regexp_replace(lower(x), '[^a-z0-9 ]', ' ', 'g'), ' +', ' ', 'g'))")
    
    # 3. Core name (remove legal suffixes)
    conn.execute("CREATE MACRO core(x) AS regexp_replace(norm(x), ' (inc|incorporated|corp|corporation|llc|ltd|limited|co|company|plc|gmbh|sa|nv|bv|srl|spa)$', '')")
    
    # 4. Numeric signature
    conn.execute("CREATE MACRO num_sig(x) AS regexp_replace(x, '[^0-9]', '', 'g')")
    
    return conn

def prepare_tables(conn, prefix, s1_path, s2_path, s3_path, limit=None):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Preparing {prefix} tables...")
    
    limit_clause = f" LIMIT {limit}" if limit else ""
    
    conn.execute(f"CREATE TABLE {prefix}_s1 AS SELECT * FROM read_csv('{s1_path}', sep='\t', header=True, ignore_errors=True){limit_clause}")
    conn.execute(f"CREATE TABLE {prefix}_s2 AS SELECT * FROM read_csv('{s2_path}', sep='\t', header=True, ignore_errors=True){limit_clause}")
    conn.execute(f"CREATE TABLE {prefix}_s3 AS SELECT * FROM read_csv('{s3_path}', sep='\t', header=True, ignore_errors=True){limit_clause}")
    
    conn.execute(f"""
        CREATE TABLE {prefix}_targets AS 
        SELECT *, 'S2' as source FROM {prefix}_s2
        UNION ALL 
        SELECT *, 'S3' as source FROM {prefix}_s3
    """)
    
    for table in [f"{prefix}_s1", f"{prefix}_targets"]:
        norm_table = f"norm_{table}"
        conn.execute(f"""
            CREATE TABLE {norm_table} AS 
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
                num_sig(business_address) as addr_num,
                -- NEW PHASE 4 FEATURES FOR JOINING
                split_part(norm(business_name), ' ', 1) as name_first_token,
                split_part(norm(business_address), ' ', 1) as addr_first_token,
                substring(alnum(business_name), 1, 6) as name_prefix,
                substring(alnum(business_address), 1, 6) as addr_prefix
            FROM {table}
        """)
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Indexing {norm_table}...")
        conn.execute(f"CREATE INDEX idx_{norm_table}_eid ON {norm_table}(entity_id)")
        conn.execute(f"CREATE INDEX idx_{norm_table}_name_norm ON {norm_table}(name_norm)")
        conn.execute(f"CREATE INDEX idx_{norm_table}_name_core ON {norm_table}(name_core)")
        conn.execute(f"CREATE INDEX idx_{norm_table}_addr_norm ON {norm_table}(addr_norm)")
        # New Phase 4 indexes
        conn.execute(f"CREATE INDEX idx_{norm_table}_nft ON {norm_table}(name_first_token)")
        conn.execute(f"CREATE INDEX idx_{norm_table}_np ON {norm_table}(name_prefix)")

def generate_candidates(conn, prefix):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Generating candidates for {prefix}...")
    
    s1 = f"norm_{prefix}_s1"
    t = f"norm_{prefix}_targets"
    
    query = f"""
        CREATE TABLE {prefix}_candidates AS 
        SELECT s1_id, t_id, min(channel) as best_channel, count(*) as num_channels
        FROM (
            -- Channel 1: Exact normalized name
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 1 as channel 
            FROM {s1} s1 JOIN {t} t ON s1.name_norm = t.name_norm WHERE s1.name_norm != ''
            
            UNION ALL
            -- Channel 2: Exact core name
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 2 as channel 
            FROM {s1} s1 JOIN {t} t ON s1.name_core = t.name_core WHERE s1.name_core != ''
            
            UNION ALL
            -- Channel 3: Exact normalized address
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 3 as channel 
            FROM {s1} s1 JOIN {t} t ON s1.addr_norm = t.addr_norm WHERE s1.addr_norm != ''
            
            UNION ALL
            -- Channel 4: Alphanumeric name + numeric address
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 4 as channel 
            FROM {s1} s1 JOIN {t} t ON s1.name_alnum = t.name_alnum AND s1.addr_num = t.addr_num 
            WHERE s1.name_alnum != '' AND s1.addr_num != ''
            
            UNION ALL
            -- Channel 5: Country + Exact core name
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 5 as channel 
            FROM {s1} s1 JOIN {t} t ON s1.country = t.country AND s1.name_core = t.name_core 
            WHERE s1.country != '' AND s1.name_core != ''
            
            UNION ALL
            -- NEW PHASE 4 CHANNELS
            -- Channel 6: First Name Token + First Addr Token + Country (Catches misspelled full names / DBAs)
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 6 as channel
            FROM {s1} s1 JOIN {t} t 
                ON s1.name_first_token = t.name_first_token 
                AND s1.addr_first_token = t.addr_first_token 
                AND s1.country = t.country
            WHERE s1.name_first_token != '' AND s1.addr_first_token != '' AND s1.country != ''
            
            UNION ALL
            -- Channel 7: Name Prefix (6 chars) + Address Prefix (6 chars) + Country
            SELECT s1.entity_id as s1_id, t.entity_id as t_id, 7 as channel
            FROM {s1} s1 JOIN {t} t 
                ON s1.name_prefix = t.name_prefix 
                AND s1.addr_prefix = t.addr_prefix
                AND s1.country = t.country
            WHERE length(s1.name_prefix) >= 5 AND length(s1.addr_prefix) >= 5 AND s1.country != ''
        )
        GROUP BY s1_id, t_id
    """
    conn.execute(query)
    
    count = conn.execute(f"SELECT COUNT(*) FROM {prefix}_candidates").fetchone()[0]
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Generated {count:,} candidate pairs.")

def extract_features_sql(conn, prefix):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Extracting features for {prefix} in SQL...")
    
    query = f"""
        CREATE TABLE {prefix}_features AS
        SELECT 
            c.s1_id, 
            c.t_id,
            c.num_channels,
            c.best_channel,
            
            CASE WHEN s1.raw_name = '' THEN 1 ELSE 0 END as s1_name_missing,
            CASE WHEN t.raw_name = '' THEN 1 ELSE 0 END as t_name_missing,
            CASE WHEN s1.raw_addr = '' THEN 1 ELSE 0 END as s1_addr_missing,
            CASE WHEN t.raw_addr = '' THEN 1 ELSE 0 END as t_addr_missing,
            
            jaro_winkler_similarity(s1.name_alnum, t.name_alnum) as name_jw,
            CASE WHEN s1.name_norm = t.name_norm AND s1.name_norm != '' THEN 1.0 ELSE 0.0 END as name_exact,
            CASE WHEN s1.name_core = t.name_core AND s1.name_core != '' THEN 1.0 ELSE 0.0 END as core_exact,
            
            jaro_winkler_similarity(s1.addr_alnum, t.addr_alnum) as addr_jw,
            CASE WHEN s1.addr_num = t.addr_num AND s1.addr_num != '' THEN 1.0 ELSE 0.0 END as num_exact,
            CASE WHEN s1.addr_num != t.addr_num AND s1.addr_num != '' AND t.addr_num != '' THEN 1.0 ELSE 0.0 END as num_contra,
            
            CASE WHEN s1.country = '' OR t.country = '' THEN 1.0 ELSE 0.0 END as country_missing,
            CASE WHEN s1.country = t.country AND s1.country != '' THEN 1.0 ELSE 0.0 END as country_agree,
            CASE WHEN s1.country != t.country AND s1.country != '' AND t.country != '' THEN 1.0 ELSE 0.0 END as country_contra
            
        FROM {prefix}_candidates c
        JOIN norm_{prefix}_s1 s1 ON c.s1_id = s1.entity_id
        JOIN norm_{prefix}_targets t ON c.t_id = t.entity_id
    """
    conn.execute(query)

def extract_train_data(conn, gt_path):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Creating training set...")
    
    conn.execute(f"CREATE TABLE train_gt (s1_id VARCHAR, matches VARCHAR)")
    conn.execute(f"COPY train_gt FROM '{gt_path}' (DELIMITER '\t', HEADER TRUE)")
    
    conn.execute("""
        CREATE TABLE train_gt_pairs AS
        SELECT s1_id, unnest(string_split(matches, ',')) as t_id
        FROM train_gt WHERE matches != ''
    """)
    
    # Label and downsample directly in SQL to prevent OOM
    conn.execute("""
        CREATE TABLE tr_labeled AS
        SELECT f.*, CASE WHEN gt.t_id IS NOT NULL THEN 1 ELSE 0 END as label
        FROM train_features f
        LEFT JOIN train_gt_pairs gt ON f.s1_id = gt.s1_id AND f.t_id = gt.t_id
    """)
    
    NEG_SAMPLE = 4_000_000
    conn.execute(f"""
        CREATE TABLE tr_sample AS
        SELECT * FROM tr_labeled WHERE label=1
        UNION ALL
        SELECT * FROM tr_labeled WHERE label=0 USING SAMPLE {NEG_SAMPLE}
    """)
    
    df = conn.execute("SELECT * FROM tr_sample").df()
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Extracted {len(df):,} training pairs.")
    return df

def run_phase4():
    run_start = time.time()
    
    db_path = str(BASE_DIR / 'experiments' / 'phase4_temp.db')
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = setup_duckdb(db_path)
    
    # 1. TRAIN MODEL
    prepare_tables(
        conn, 'train', 
        DATA_DIR / 'train' / 'train_source1.tsv',
        DATA_DIR / 'train' / 'train_source2.tsv',
        DATA_DIR / 'train' / 'train_source3.tsv',
        limit=600000
    )
    generate_candidates(conn, 'train')
    extract_features_sql(conn, 'train')
    train_df = extract_train_data(conn, DATA_DIR / 'train' / 'train_ground_truth.tsv')
    
    features = [
        'num_channels', 'best_channel', 
        's1_name_missing', 't_name_missing', 's1_addr_missing', 't_addr_missing',
        'name_jw', 'name_exact', 'core_exact',
        'addr_jw', 'num_exact', 'num_contra',
        'country_missing', 'country_agree', 'country_contra'
    ]
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training LightGBM on {len(train_df):,} pairs...")
    X_train = train_df[features].fillna(0)
    y_train = train_df['label']
    
    # Increased n_estimators slightly to handle harder negatives from new channels
    model = LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model trained. Positive samples: {y_train.sum():,}")
    
    conn.execute("DROP TABLE train_s1; DROP TABLE train_s2; DROP TABLE train_s3; DROP TABLE train_targets;")
    conn.execute("DROP TABLE norm_train_s1; DROP TABLE norm_train_targets;")
    conn.execute("DROP TABLE train_candidates; DROP TABLE train_features; DROP TABLE tr_labeled; DROP TABLE tr_sample;")
    
    # 2. FULL TEST INFERENCE
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] === STARTING FULL TEST INFERENCE ===")
    prepare_tables(
        conn, 'test', 
        DATA_DIR / 'test' / 'test_source1.tsv',
        DATA_DIR / 'test' / 'test_source2.tsv',
        DATA_DIR / 'test' / 'test_source3.tsv'
    )
    
    generate_candidates(conn, 'test')
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Writing candidate_pairs.tsv (Official output 1)...")
    conn.execute(f"""
        COPY (
            SELECT s1.entity_id as source1_entity_id, 
                   string_agg(c.t_id, ',') as candidate_entity_ids
            FROM test_s1 s1
            LEFT JOIN test_candidates c ON s1.entity_id = c.s1_id
            GROUP BY s1.entity_id
        ) TO '{OUT_DIR / "candidate_pairs.tsv"}' (HEADER, DELIMITER '\t')
    """)
    
    extract_features_sql(conn, 'test')
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Predicting full test set with LightGBM in chunks...")
    chunk_size = 1000000
    offset = 0
    predictions = []
    while True:
        chunk = conn.execute(f"SELECT s1_id, t_id, {','.join(features)} FROM test_features LIMIT {chunk_size} OFFSET {offset}").df()
        if len(chunk) == 0:
            break
            
        X_test = chunk[features].fillna(0)
        chunk['score'] = model.predict_proba(X_test)[:, 1]
        
        matches = chunk[chunk['score'] > 0.5][['s1_id', 't_id', 'score']]
        predictions.append(matches)
        
        offset += chunk_size
        print(f"  Scored {offset:,} candidate pairs...")
        
    pred_df = pd.concat(predictions) if predictions else pd.DataFrame(columns=['s1_id', 't_id', 'score'])
        
    conn.execute("CREATE TABLE test_preds AS SELECT * FROM pred_df")
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Writing matching_results.tsv (Official output 2)...")
    conn.execute(f"""
        COPY (
            SELECT s1.entity_id as source1_entity_id,
                   coalesce(string_agg(p.t_id, ',' ORDER BY p.score DESC), '') as matched_entity_ids
            FROM test_s1 s1
            LEFT JOIN test_preds p ON s1.entity_id = p.s1_id
            GROUP BY s1.entity_id
        ) TO '{OUT_DIR / "matching_results.tsv"}' (HEADER, DELIMITER '\t')
    """)
    
    runtime = time.time() - run_start
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] === PHASE 4 COMPLETE in {runtime:.0f}s ===")
    
    # 3. STATS & SNAPSHOT
    s1_count = conn.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]
    cand_count = conn.execute("SELECT COUNT(*) FROM test_candidates").fetchone()[0]
    match_count = len(pred_df)
    empty_cands = conn.execute("""
        SELECT COUNT(*) FROM test_s1 s1 
        LEFT JOIN test_candidates c ON s1.entity_id = c.s1_id 
        WHERE c.t_id IS NULL
    """).fetchone()[0]
    
    stats = {
        "test_s1_count": s1_count,
        "total_candidate_pairs": cand_count,
        "total_matches": match_count,
        "empty_candidate_s1": empty_cands,
        "runtime_seconds": int(runtime),
        "architecture": "DuckDB Native Pipeline + Phase 4 Enhanced Recall",
        "channels": [
            "Exact name_norm", "Exact name_core", "Exact addr_norm",
            "name_alnum + addr_num", "country + name_core",
            "first_name_token + first_addr_token + country",
            "name_prefix(6) + addr_prefix(6) + country"
        ]
    }
    
    with open(SUB_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    shutil.copy(OUT_DIR / "matching_results.tsv", SUB_DIR / "matching_results.tsv")
    shutil.copy(OUT_DIR / "candidate_pairs.tsv",  SUB_DIR / "candidate_pairs.tsv")

if __name__ == "__main__":
    run_phase4()
