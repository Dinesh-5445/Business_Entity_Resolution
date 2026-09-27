import csv
import os
import collections
import json
from pathlib import Path

# Paths
BASE_DIR = Path(r"c:\Users\Palav\Downloads\BUSINESS_ENTITY-RESOLUTION")
TRAIN_DIR = BASE_DIR / "student_resource" / "dataset" / "train"
TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"
OUTPUT_FILE = BASE_DIR / "docs" / "DATA_PROFILE.md"

def analyze_file(filepath):
    print(f"Analyzing {filepath.name}...")
    stats = {
        "row_count": 0,
        "unique_ids": set(),
        "duplicate_ids_count": 0,
        "missing_values": collections.defaultdict(int),
        "empty_strings": collections.defaultdict(int),
        "field_lengths": collections.defaultdict(list),
        "unique_names": set(),
        "repeated_names_count": 0,
        "unique_addresses": set(),
        "repeated_addresses_count": 0,
        "unique_name_address_combos": set(),
        "repeated_name_address_count": 0,
        "country_distribution": collections.defaultdict(int),
        "number_tokens_dist": collections.defaultdict(int)
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            headers = reader.fieldnames
            if not headers:
                return stats
            
            for row in reader:
                stats["row_count"] += 1
                
                # Check IDs
                entity_id = row.get("entity_id", "")
                if entity_id in stats["unique_ids"]:
                    stats["duplicate_ids_count"] += 1
                else:
                    stats["unique_ids"].add(entity_id)
                
                # Check missing/empty
                for field in headers:
                    val = row.get(field)
                    if val is None:
                        stats["missing_values"][field] += 1
                    elif str(val).strip() == "":
                        stats["empty_strings"][field] += 1
                    else:
                        stats["field_lengths"][field].append(len(str(val)))
                
                name = row.get("business_name", "")
                address = row.get("business_address", "")
                country = row.get("country", "")
                
                # Names
                if name:
                    if name in stats["unique_names"]:
                        stats["repeated_names_count"] += 1
                    else:
                        stats["unique_names"].add(name)
                
                # Addresses
                if address:
                    if address in stats["unique_addresses"]:
                        stats["repeated_addresses_count"] += 1
                    else:
                        stats["unique_addresses"].add(address)
                
                # Combos
                if name and address:
                    combo = f"{name} || {address}"
                    if combo in stats["unique_name_address_combos"]:
                        stats["repeated_name_address_count"] += 1
                    else:
                        stats["unique_name_address_combos"].add(combo)
                
                # Country
                if country:
                    stats["country_distribution"][country] += 1
                    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    # Free up sets to save memory if large
    stats["unique_ids_count"] = len(stats["unique_ids"])
    del stats["unique_ids"]
    stats["unique_names_count"] = len(stats["unique_names"])
    del stats["unique_names"]
    stats["unique_addresses_count"] = len(stats["unique_addresses"])
    del stats["unique_addresses"]
    stats["unique_name_address_combos_count"] = len(stats["unique_name_address_combos"])
    del stats["unique_name_address_combos"]
    
    return stats

def analyze_ground_truth(filepath):
    print(f"Analyzing {filepath.name}...")
    stats = {
        "row_count": 0,
        "positive_match_cardinality": collections.defaultdict(int),
        "s1_matching_s2_count": 0,
        "s1_matching_s3_count": 0,
        "singleton_s1_count": 0,
        "target_appearances": collections.defaultdict(int)
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            # Header typically source1_entity_id, matched_entity_ids (comma separated)
            header = next(reader, None)
            for row in reader:
                if len(row) < 1: continue
                stats["row_count"] += 1
                s1_id = row[0]
                matched_ids_str = row[1] if len(row) > 1 else ""
                
                if not matched_ids_str.strip():
                    stats["singleton_s1_count"] += 1
                    stats["positive_match_cardinality"][0] += 1
                    continue
                
                matched_ids = [m.strip() for m in matched_ids_str.split(',') if m.strip()]
                stats["positive_match_cardinality"][len(matched_ids)] += 1
                
                has_s2 = False
                has_s3 = False
                
                for m_id in matched_ids:
                    stats["target_appearances"][m_id] += 1
                    if "-S2-" in m_id or m_id.startswith("S2"): 
                        has_s2 = True
                    elif "-S3-" in m_id or m_id.startswith("S3"):
                        has_s3 = True
                
                if has_s2: stats["s1_matching_s2_count"] += 1
                if has_s3: stats["s1_matching_s3_count"] += 1
                
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return stats

def generate_report():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    files_to_analyze = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv"
    ]
    
    report_lines = ["# Data Profile\n"]
    
    for fpath in files_to_analyze:
        if fpath.exists():
            stats = analyze_file(fpath)
            report_lines.append(f"## {fpath.name}\n")
            report_lines.append(f"- **Row count:** {stats['row_count']}")
            report_lines.append(f"- **Unique IDs:** {stats['unique_ids_count']}")
            report_lines.append(f"- **Duplicate IDs:** {stats['duplicate_ids_count']}")
            report_lines.append(f"- **Missing values:** {dict(stats['missing_values'])}")
            report_lines.append(f"- **Empty strings:** {dict(stats['empty_strings'])}")
            
            # Avg lengths
            lengths = {}
            for k, v in stats['field_lengths'].items():
                if v:
                    lengths[k] = {"min": min(v), "max": max(v), "avg": sum(v)//len(v)}
            report_lines.append(f"- **Field lengths:** {lengths}")
            
            report_lines.append(f"- **Unique names:** {stats['unique_names_count']} (Repeated: {stats['repeated_names_count']})")
            report_lines.append(f"- **Unique addresses:** {stats['unique_addresses_count']} (Repeated: {stats['repeated_addresses_count']})")
            report_lines.append(f"- **Unique name+addr combos:** {stats['unique_name_address_combos_count']} (Repeated: {stats['repeated_name_address_count']})")
            
            # Country distribution top 10
            top_countries = collections.Counter(stats['country_distribution']).most_common(10)
            report_lines.append(f"- **Top Countries:** {top_countries}")
            report_lines.append("\n")
            
    # Ground truth
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    if gt_path.exists():
        gt_stats = analyze_ground_truth(gt_path)
        report_lines.append(f"## Ground Truth ({gt_path.name})\n")
        report_lines.append(f"- **Total S1 records in GT:** {gt_stats['row_count']}")
        report_lines.append(f"- **Singleton S1 count (zero matches):** {gt_stats['singleton_s1_count']}")
        report_lines.append(f"- **S1 matching S2:** {gt_stats['s1_matching_s2_count']}")
        report_lines.append(f"- **S1 matching S3:** {gt_stats['s1_matching_s3_count']}")
        
        report_lines.append(f"- **Positive match cardinality distribution:** {dict(gt_stats['positive_match_cardinality'])}")
        
        # Target appearances
        mult_targets = sum(1 for v in gt_stats['target_appearances'].values() if v > 1)
        report_lines.append(f"- **S2/S3 targets appearing in multiple S1 ground truth lists:** {mult_targets}")
        report_lines.append("\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Report written to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
