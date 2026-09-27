import csv
from pathlib import Path
import json

BASE_DIR = Path('c:/Users/Palav/Downloads/BUSINESS_ENTITY-RESOLUTION')
OUT_DIR = BASE_DIR / 'output'
SUB_DIR = BASE_DIR / 'submissions' / 'phase_3_full_baseline'

# Ensure all 1,732,544 rows exist in the output files
s1_ids = []
with open(BASE_DIR / 'student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        s1_ids.append(row['entity_id'])

def fix_file(filename):
    print(f"Fixing {filename}...")
    existing = {}
    with open(OUT_DIR / filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if not row: continue
            # Handle DuckDB's weird empty quotes
            vals = row[1] if len(row) > 1 else ""
            if vals == '""' or vals == '"': vals = ""
            existing[row[0]] = vals

    with open(OUT_DIR / filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if "candidate" in filename:
            writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        else:
            writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        
        for s1 in s1_ids:
            writer.writerow([s1, existing.get(s1, "")])

fix_file('candidate_pairs.tsv')
fix_file('matching_results.tsv')

print("Done fixing TSVs.")
