#!/usr/bin/env python3
"""
Anomaly Detection Evaluation - Segment-Based Evaluation
If ANY point in a continuous abnormal segment is detected, 
the entire segment is considered correctly detected.
Based on: Xu et al., 2018; Su et al., 2019; Shen et al., 2020
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys


def load_ground_truth(test_dataset_csv):
    """Load ground truth labels from test dataset"""
    print(f"Loading ground truth from: {test_dataset_csv}")
    
    with open(test_dataset_csv, 'r') as f:
        lines = f.readlines()
    
    labels = []
    true_count = 0
    
    # Skip header
    for idx, line in enumerate(lines[1:]):
        line = line.rstrip().rstrip(',').rstrip()
        if not line:
            continue
        
        # Check if line ends with ,True
        if line.endswith(',True') or line.endswith(', True'):
            labels.append(1)
            true_count += 1
        else:
            labels.append(0)
    
    print(f"  Total lines: {len(lines)}")
    print(f"  Data rows: {len(labels)}")
    print(f"  Anomaly points (marked True): {true_count}")
    print(f"  Normal: {len(labels) - true_count}")
    
    return np.array(labels)


def load_predictions(detection_csv):
    """Load anomaly detection predictions"""
    print(f"Loading predictions from: {detection_csv}")
    
    df = pd.read_csv(detection_csv)
    
    # Find the anomaly prediction column
    anomaly_col = None
    for col in ['is_anomaly', 'anomaly', 'prediction', 'pred', 'detected']:
        if col in df.columns:
            anomaly_col = col
            break
    
    if anomaly_col is None:
        print(f"❌ Error: Could not find anomaly prediction column")
        print(f"   Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    predictions = df[anomaly_col].values.astype(int)
    
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values
    else:
        timestamps = np.arange(len(predictions))
    
    if 'anomaly_score' in df.columns:
        scores = df['anomaly_score'].values
    else:
        scores = np.zeros(len(predictions))
    
    print(f"  Loaded {len(predictions)} predictions")
    print(f"  Detected anomalies: {sum(predictions)}")
    print(f"  Normal predictions: {len(predictions) - sum(predictions)}")
    
    return predictions, timestamps, scores


def find_segments(gt):
    """
    Find continuous segments of anomalies in ground truth
    Returns list of (start_idx, end_idx) for each segment
    """
    segments = []
    in_segment = False
    start = 0
    
    for i in range(len(gt)):
        if gt[i] == 1 and not in_segment:
            # Start new segment
            in_segment = True
            start = i
        elif gt[i] == 0 and in_segment:
            # End current segment
            in_segment = False
            segments.append((start, i - 1))
    
    # Handle segment that extends to end
    if in_segment:
        segments.append((start, len(gt) - 1))
    
    return segments


def segment_based_evaluation(gt, pred):
    """
    Segment-based evaluation:
    If ANY point in a continuous abnormal segment is detected,
    the entire segment is correctly detected.
    """
    # Find all continuous segments in ground truth
    segments = find_segments(gt)
    
    print(f"\nSegment-based Evaluation")
    print(f"  Found {len(segments)} continuous anomalous segments in GT")
    
    if len(segments) > 0:
        print(f"  Segment details:")
        for idx, (start, end) in enumerate(segments):
            length = end - start + 1
            has_detection = np.any(pred[start:end+1] == 1)
            status = "✓ DETECTED" if has_detection else "✗ MISSED"
            print(f"    Segment {idx+1}: [{start:4d}-{end:4d}] (length {length:2d}) {status}")
    
    # Count segments with at least one detection
    detected_segments = 0
    for start, end in segments:
        if np.any(pred[start:end+1] == 1):
            detected_segments += 1
    
    print(f"  Detected segments: {detected_segments}/{len(segments)}")
    
    # Create adjusted predictions: mark entire segment as predicted if ANY point detected
    pred_adjusted = pred.copy()
    for start, end in segments:
        if np.any(pred[start:end+1] == 1):
            pred_adjusted[start:end+1] = 1
    
    return pred_adjusted, segments, detected_segments


def print_metrics(gt, pred, label=""):
    """Print evaluation metrics"""
    if label:
        print(f"\n{label}")
        print("-" * 70)
    
    print(f"pred: {pred.shape}")
    print(f"gt:   {gt.shape}")
    
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        gt, pred, average='binary'
    )
    
    print(
        f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, "
        f"Recall : {recall:.4f}, F-score : {f_score:.4f}"
    )
    
    # Confusion matrix
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    return accuracy, precision, recall, f_score


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Anomaly Detection Evaluation - Segment-Based (Xu et al., 2018)'
    )
    parser.add_argument('test_dataset', help='Test dataset CSV with ground truth labels')
    parser.add_argument('detection_results', help='Detection results CSV from anomaly detector')
    parser.add_argument('--output', help='Save results to CSV file')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ANOMALY DETECTION EVALUATION - SEGMENT-BASED")
    print("="*70)
    print("Strategy: If ANY point in a continuous abnormal segment is detected,")
    print("the entire segment is considered correctly detected.")
    print("(Xu et al., 2018; Su et al., 2019; Shen et al., 2020)")
    
    # Load data
    gt = load_ground_truth(args.test_dataset)
    pred, timestamps, scores = load_predictions(args.detection_results)
    
    # Match sizes
    if len(gt) != len(pred):
        print(f"\n⚠ Warning: Ground truth has {len(gt)} samples but predictions have {len(pred)}")
        min_len = min(len(gt), len(pred))
        gt = gt[:min_len]
        pred = pred[:min_len]
        timestamps = timestamps[:min_len]
        scores = scores[:min_len]
        print(f"  Using first {min_len} samples for evaluation")
    
    # Raw point-based evaluation
    print("\n" + "="*70)
    print("POINT-BASED EVALUATION (Strict - Exact Point Matching)")
    print("="*70)
    accuracy_point, precision_point, recall_point, f_score_point = print_metrics(
        gt, pred, "RAW RESULTS"
    )
    
    # Segment-based evaluation
    print("\n" + "="*70)
    print("SEGMENT-BASED EVALUATION (If ANY point detected -> segment OK)")
    print("="*70)
    pred_adjusted, segments, detected_segments = segment_based_evaluation(gt, pred)
    accuracy_seg, precision_seg, recall_seg, f_score_seg = print_metrics(
        gt, pred_adjusted, "RESULTS AFTER SEGMENT ADJUSTMENT"
    )
    
    # Comparison
    print("\n" + "="*70)
    print("IMPROVEMENT FROM SEGMENT-BASED EVALUATION")
    print("="*70)
    print(f"Accuracy:  {accuracy_point:.4f} -> {accuracy_seg:.4f} ({(accuracy_seg-accuracy_point):+.4f})")
    print(f"Precision: {precision_point:.4f} -> {precision_seg:.4f} ({(precision_seg-precision_point):+.4f})")
    print(f"Recall:    {recall_point:.4f} -> {recall_seg:.4f} ({(recall_seg-recall_point):+.4f})")
    print(f"F-score:   {f_score_point:.4f} -> {f_score_seg:.4f} ({(f_score_seg-f_score_point):+.4f})")
    
    # Save if requested
    if args.output:
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'ground_truth': gt,
            'pred_raw': pred,
            'pred_segment_based': pred_adjusted,
            'anomaly_score': scores
        })
        results_df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")
    
    print("\n" + "="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())