import numpy as np

def compute_s1_metrics(y_true, y_pred):
    """
    Computes Precision, Recall, and F0.5 for a single Source 1 entity.
    
    Args:
        y_true (set or list): Ground truth matched entity IDs.
        y_pred (set or list): Predicted matched entity IDs.
        
    Returns:
        dict: {'precision': float, 'recall': float, 'f0_5': float, 
               'tp': int, 'fp': int, 'fn': int}
    """
    y_true = set(y_true)
    y_pred = set(y_pred)
    
    # If both are empty (true singleton and predicted singleton)
    if not y_true and not y_pred:
        return {'precision': 1.0, 'recall': 1.0, 'f0_5': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
    # If ground truth is empty but we predicted something (false positive singleton)
    if not y_true and y_pred:
        return {'precision': 0.0, 'recall': 0.0, 'f0_5': 0.0, 'tp': 0, 'fp': len(y_pred), 'fn': 0}
        
    # If ground truth has matches but we predicted nothing (false negative singleton)
    if y_true and not y_pred:
        return {'precision': 0.0, 'recall': 0.0, 'f0_5': 0.0, 'tp': 0, 'fp': 0, 'fn': len(y_true)}
        
    tp = len(y_true.intersection(y_pred))
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        # F0.5 = (1.25 * P * R) / (0.25 * P + R)
        f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    else:
        f0_5 = 0.0
        
    return {
        'precision': precision,
        'recall': recall,
        'f0_5': f0_5,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def compute_macro_f0_5(ground_truth_dict, prediction_dict):
    """
    Computes macro-averaged metrics across all Source 1 entities.
    
    Args:
        ground_truth_dict (dict): Mapping {s1_id: [matched_ids]}
        prediction_dict (dict): Mapping {s1_id: [predicted_ids]}
        
    Returns:
        dict: {'macro_f0_5': float, 'macro_precision': float, 'macro_recall': float}
    """
    all_s1_ids = set(ground_truth_dict.keys())
    # Note: Predictions might have extra or missing keys, usually evaluate on ground truth keys
    
    metrics_list = []
    
    for s1_id in all_s1_ids:
        y_true = ground_truth_dict.get(s1_id, [])
        y_pred = prediction_dict.get(s1_id, [])
        metrics = compute_s1_metrics(y_true, y_pred)
        metrics_list.append(metrics)
        
    macro_precision = np.mean([m['precision'] for m in metrics_list])
    macro_recall = np.mean([m['recall'] for m in metrics_list])
    macro_f0_5 = np.mean([m['f0_5'] for m in metrics_list])
    
    # Also sum TP, FP, FN
    total_tp = sum(m['tp'] for m in metrics_list)
    total_fp = sum(m['fp'] for m in metrics_list)
    total_fn = sum(m['fn'] for m in metrics_list)
    
    return {
        'macro_f0_5': macro_f0_5,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'num_entities': len(all_s1_ids)
    }
