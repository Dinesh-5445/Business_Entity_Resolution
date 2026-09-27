import csv
import collections
import numpy as np

def build_connected_components(ground_truth_file):
    """
    Reads ground truth and builds connected components of S1 entities.
    Two S1 entities are in the same component if they share a matched S2/S3 entity,
    or transitively.
    
    Args:
        ground_truth_file (str): Path to train_ground_truth.tsv
        
    Returns:
        list of list: Each inner list contains S1 IDs belonging to the same component.
    """
    s1_to_targets = collections.defaultdict(list)
    target_to_s1s = collections.defaultdict(list)
    
    # Read graph
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        for row in reader:
            if not row: continue
            s1_id = row[0]
            if len(row) > 1 and row[1].strip():
                targets = [t.strip() for t in row[1].split(',')]
                s1_to_targets[s1_id].extend(targets)
                for t in targets:
                    target_to_s1s[t].append(s1_id)
            else:
                # Singleton S1
                s1_to_targets[s1_id] = []
                
    all_s1_ids = list(s1_to_targets.keys())
    visited_s1 = set()
    components = []
    
    for s1 in all_s1_ids:
        if s1 not in visited_s1:
            # BFS/DFS to find component
            queue = [s1]
            visited_s1.add(s1)
            comp = []
            
            while queue:
                curr_s1 = queue.pop(0)
                comp.append(curr_s1)
                
                # Get targets
                for target in s1_to_targets[curr_s1]:
                    # Get S1s sharing this target
                    for neighbor_s1 in target_to_s1s[target]:
                        if neighbor_s1 not in visited_s1:
                            visited_s1.add(neighbor_s1)
                            queue.append(neighbor_s1)
            
            components.append(comp)
            
    return components

def create_folds(components, num_folds=5, seed=42):
    """
    Distributes components into k folds.
    
    Args:
        components (list of list): Connected components.
        num_folds (int): Number of folds.
        seed (int): Random seed.
        
    Returns:
        dict: {fold_idx: [s1_ids]}
    """
    np.random.seed(seed)
    
    # Shuffle components to ensure random distribution
    # Sort first to ensure determinism across runs before shuffling
    components_sorted = [sorted(comp) for comp in components]
    components_sorted.sort(key=lambda x: (len(x), x[0]))
    
    np.random.shuffle(components_sorted)
    
    folds = {i: [] for i in range(num_folds)}
    fold_sizes = {i: 0 for i in range(num_folds)}
    
    # Greedy distribution to balance fold sizes (number of S1s)
    for comp in components_sorted:
        # Find fold with minimum size
        smallest_fold = min(fold_sizes, key=fold_sizes.get)
        folds[smallest_fold].extend(comp)
        fold_sizes[smallest_fold] += len(comp)
        
    return folds

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        gt_file = sys.argv[1]
        comps = build_connected_components(gt_file)
        print(f"Total components: {len(comps)}")
        folds = create_folds(comps, num_folds=5)
        for i, fold_s1s in folds.items():
            print(f"Fold {i} size: {len(fold_s1s)} S1 entities")
