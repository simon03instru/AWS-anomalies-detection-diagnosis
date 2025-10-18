import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from datetime import datetime
import sys
import argparse

def load_ground_truth(csv_path):
    """Load ground truth dataset"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_predictions(csv_path):
    """Load anomaly detection predictions"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def align_datasets(gt_df, pred_df):
    """
    Align ground truth and predictions by row index
    Returns: gt_labels, pred_labels (both as numpy arrays)
    """
    min_len = min(len(gt_df), len(pred_df))
    
    gt_labels = gt_df['is_anomaly'].values[:min_len].astype(int)
    pred_labels = pred_df['is_anomaly'].values[:min_len].astype(int)
    
    print(f"✓ Aligned {len(gt_labels)} data points by row")
    print(f"  Ground truth anomalies: {np.sum(gt_labels)}")
    print(f"  Predicted anomalies: {np.sum(pred_labels)}")
    
    return gt_labels, pred_labels

def apply_adjustment_strategy(gt, pred):
    """
    Apply adjustment strategy: if any point in an anomalous segment is detected,
    mark entire segment as detected.
    
    Args:
        gt: numpy array of ground truth labels (0 or 1)
        pred: numpy array of predictions (0 or 1)
    
    Returns:
        adjusted_pred: numpy array with adjustment applied
    """
    adjusted_pred = pred.copy()
    anomaly_state = False
    
    for i in range(len(gt)):
        # Trigger adjustment when we find a correct detection at segment start
        if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            
            # Backward fill: fill missed points before current position
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
            
            # Forward fill: fill missed points after current position
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
        
        # Exit anomaly state when we reach normal region
        elif gt[i] == 0:
            anomaly_state = False
        
        # Continue marking as anomaly while in segment
        if anomaly_state:
            adjusted_pred[i] = 1
    
    return adjusted_pred

def apply_lag_tolerance(gt, pred, lag_tolerance=1):
    """
    Apply lag tolerance: Convert predictions that occur within lag_tolerance timesteps
    of a ground truth anomaly into correct detections.
    
    This handles:
    - Lagged detections after anomaly ends (t+1, t+2, etc.)
    - Early detections before anomaly starts (t-1, t-2, etc.)
    - Detections within anomalous segments
    
    Args:
        gt: ground truth labels
        pred: predictions
        lag_tolerance: number of timesteps to allow detection delay/early (default: 1)
    
    Returns:
        adjusted_pred: predictions adjusted for lag tolerance
    """
    adjusted_pred = pred.copy()
    
    # Find all GT anomalous segments
    gt_segments = []
    i = 0
    while i < len(gt):
        if gt[i] == 1:
            start = i
            while i < len(gt) and gt[i] == 1:
                i += 1
            end = i - 1
            gt_segments.append((start, end))
        else:
            i += 1
    
    # For each GT segment, check predictions within tolerance window
    for seg_start, seg_end in gt_segments:
        # Define detection window: from (seg_start - lag_tolerance) to (seg_end + lag_tolerance)
        window_start = max(0, seg_start - lag_tolerance)
        window_end = min(len(gt) - 1, seg_end + lag_tolerance)
        
        # Check if any prediction exists in this window
        detection_found = False
        for j in range(window_start, window_end + 1):
            if pred[j] == 1:
                detection_found = True
                break
        
        # If detection found within lag window, mark entire segment as detected
        if detection_found:
            for j in range(seg_start, seg_end + 1):
                adjusted_pred[j] = 1
            
            # Also mark lagged/early detections as part of anomaly (convert FP to TP)
            for j in range(window_start, window_end + 1):
                if pred[j] == 1 and (j < seg_start or j > seg_end):
                    # This is a lagged or early detection - mark it as TP
                    adjusted_pred[j] = 1
    
    return adjusted_pred

def calculate_metrics(gt, pred_adjusted, lag_tolerance=1):
    """Calculate and print metrics, treating lagged predictions as correct"""
    
    # First, identify all GT segments
    gt_segments = []
    i = 0
    while i < len(gt):
        if gt[i] == 1:
            start = i
            while i < len(gt) and gt[i] == 1:
                i += 1
            end = i - 1
            gt_segments.append((start, end))
        else:
            i += 1
    
    # Create a modified prediction array that treats lagged detections as correct
    pred_modified = pred_adjusted.copy()
    
    # Find which predictions in normal regions are actually lagged detections
    lagged_detections = []
    
    for i in range(len(pred_modified)):
        if pred_modified[i] == 1 and gt[i] == 0:  # Prediction in normal region
            # Check if this is a lagged detection near a GT segment
            for seg_start, seg_end in gt_segments:
                distance_to_segment = min(abs(i - seg_start), abs(i - seg_end))
                
                # If within lag tolerance of a segment, this is a valid detection
                if distance_to_segment <= lag_tolerance:
                    # Mark corresponding segment as detected
                    for j in range(seg_start, seg_end + 1):
                        pred_modified[j] = 1
                    # Remove this prediction from normal region (it's part of anomaly detection)
                    pred_modified[i] = 0
                    lagged_detections.append(i)
                    break
    
    # Confusion matrix on modified predictions
    tn, fp, fn, tp = confusion_matrix(gt, pred_modified).ravel()
    
    # Metrics
    accuracy = accuracy_score(gt, pred_modified)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, pred_modified, average='binary'
    )
    
    # Find indices of errors on MODIFIED predictions
    fn_indices = np.where((gt == 1) & (pred_modified == 0))[0]
    fp_indices = np.where((gt == 0) & (pred_modified == 1))[0]
    tp_indices = np.where((gt == 1) & (pred_modified == 1))[0]
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f_score': f_score,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'fn_indices': fn_indices,
        'fp_indices': fp_indices,
        'tp_indices': tp_indices,
        'lagged_detections': np.array(lagged_detections),
        'gt_segments': gt_segments,
        'pred_modified': pred_modified
    }

def print_detailed_report(gt, pred_original, pred_adjusted, metrics, lag_tolerance=1):
    """Print detailed evaluation report"""
    print("\n" + "="*70)
    print("ANOMALY DETECTION EVALUATION REPORT")
    print("="*70)
    
    print(f"\n[0] CONFIGURATION")
    print(f"    Lag tolerance: {lag_tolerance} timestep(s)")
    print(f"    Total data points: {len(gt)}")
    
    print("\n[1] CONFUSION MATRIX BREAKDOWN")
    print(f"    True Positives (TP):  {metrics['tp']:6d}  (Correctly detected anomalies)")
    print(f"    True Negatives (TN):  {metrics['tn']:6d}  (Correctly identified normal)")
    print(f"    False Positives (FP): {metrics['fp']:6d}  (False alarms)")
    print(f"    False Negatives (FN): {metrics['fn']:6d}  (Missed anomalies)")
    
    print("\n[2] PERFORMANCE METRICS")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}  (Overall correctness)")
    print(f"    Precision: {metrics['precision']:.4f}  (Quality of positive predictions)")
    print(f"    Recall:    {metrics['recall']:.4f}  (Coverage of true anomalies)")
    print(f"    F-score:   {metrics['f_score']:.4f}  (Harmonic mean of precision & recall)")
    
    print("\n[3] ADJUSTMENT STRATEGY & LAG TOLERANCE IMPACT")
    changes = np.sum(pred_original != pred_adjusted)
    lagged_count = len(metrics['lagged_detections'])
    print(f"    Original predictions: {np.sum(pred_original)} anomalies flagged")
    print(f"    After lag tolerance: {np.sum(pred_original)} anomalies flagged")
    print(f"    After adjustment:    {np.sum(pred_adjusted)} anomalies flagged")
    print(f"    Points modified:     {changes}")
    print(f"    Lagged detections handled: {lagged_count}")
    
    print("\n[4] FALSE NEGATIVES (MISSED ANOMALIES)")
    if len(metrics['fn_indices']) > 0:
        print(f"    Count: {len(metrics['fn_indices'])}")
        print(f"    Indices: {metrics['fn_indices'].tolist()}")
        
        # Group consecutive indices into segments
        fn_segments = []
        if len(metrics['fn_indices']) > 0:
            start = metrics['fn_indices'][0]
            end = metrics['fn_indices'][0]
            for idx in metrics['fn_indices'][1:]:
                if idx == end + 1:
                    end = idx
                else:
                    fn_segments.append((start, end))
                    start = idx
                    end = idx
            fn_segments.append((start, end))
        
        print(f"    Segments (start-end): {fn_segments}")
    else:
        print(f"    Count: 0 ✓ (No missed anomalies)")
    
    print("\n[5] FALSE POSITIVES (FALSE ALARMS)")
    if len(metrics['fp_indices']) > 0:
        print(f"    Count: {len(metrics['fp_indices'])}")
        print(f"    Indices: {metrics['fp_indices'].tolist()}")
        
        # Group consecutive indices into segments
        fp_segments = []
        if len(metrics['fp_indices']) > 0:
            start = metrics['fp_indices'][0]
            end = metrics['fp_indices'][0]
            for idx in metrics['fp_indices'][1:]:
                if idx == end + 1:
                    end = idx
                else:
                    fp_segments.append((start, end))
                    start = idx
                    end = idx
            fp_segments.append((start, end))
        
        print(f"    Segments (start-end): {fp_segments}")
    else:
        print(f"    Count: 0 ✓ (No false alarms)")
    
    if len(metrics['lagged_detections']) > 0:
        print(f"\n    ✓ LAGGED DETECTIONS CORRECTED: {len(metrics['lagged_detections'])}")
        print(f"      Indices: {metrics['lagged_detections'].tolist()}")
        print(f"      These were converted from FP to TP (lagged detection at t+{lag_tolerance})")
    
    print("\n[6] INTERPRETATION")
    if metrics['recall'] == 1.0:
        print("    ✓ EXCELLENT: All true anomalies detected (no false negatives)")
    elif metrics['recall'] > 0.9:
        print("    ✓ GOOD: Most anomalies detected")
    elif metrics['recall'] > 0.7:
        print("    ⚠ FAIR: Some anomalies missed")
    else:
        print("    ✗ POOR: Many anomalies missed")
    
    if metrics['precision'] == 1.0:
        print("    ✓ EXCELLENT: No false alarms (perfect precision)")
    elif metrics['precision'] > 0.9:
        print("    ✓ GOOD: Few false alarms")
    elif metrics['precision'] > 0.7:
        print("    ⚠ FAIR: Some false alarms")
    else:
        print("    ✗ POOR: Many false alarms")
    
    print("\n" + "="*70)

def main(gt_path, pred_path, lag_tolerance=1):
    print("Loading datasets...")
    try:
        gt_df = load_ground_truth(gt_path)
        pred_df = load_predictions(pred_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    
    print(f"✓ Ground truth loaded: {len(gt_df)} records from {gt_path}")
    print(f"✓ Predictions loaded: {len(pred_df)} records from {pred_path}")
    
    # Align datasets
    print("\nAligning datasets...")
    gt_labels, pred_labels = align_datasets(gt_df, pred_df)
    
    # Apply lag tolerance first
    print(f"\nApplying lag tolerance ({lag_tolerance} timesteps)...")
    pred_with_lag = apply_lag_tolerance(gt_labels, pred_labels, lag_tolerance)
    
    # Apply adjustment strategy
    print("Applying adjustment strategy...")
    pred_adjusted = apply_adjustment_strategy(gt_labels, pred_with_lag)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(gt_labels, pred_adjusted, lag_tolerance)
    
    # Print detailed report
    print_detailed_report(gt_labels, pred_labels, pred_adjusted, metrics, lag_tolerance)
    
    # Return results for further use
    return {
        'ground_truth': gt_labels,
        'predictions_original': pred_labels,
        'predictions_with_lag': pred_with_lag,
        'predictions_adjusted': pred_adjusted,
        'metrics': metrics,
        'gt_df': gt_df,
        'pred_df': pred_df,
        'lag_tolerance': lag_tolerance
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate anomaly detection predictions against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py ground_truth.csv predictions.csv
  python script.py ground_truth.csv predictions.csv --lag-tolerance 2
        """
    )
    
    parser.add_argument(
        'ground_truth',
        help='Path to ground truth CSV file'
    )
    parser.add_argument(
        'predictions',
        help='Path to anomaly predictions CSV file'
    )
    parser.add_argument(
        '--lag-tolerance',
        type=int,
        default=1,
        help='Number of timesteps to allow detection delay (default: 1)'
    )
    
    args = parser.parse_args()
    
    results = main(args.ground_truth, args.predictions, args.lag_tolerance)