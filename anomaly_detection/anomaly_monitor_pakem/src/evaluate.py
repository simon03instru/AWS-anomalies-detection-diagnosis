import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                            f1_score, roc_curve, auc, roc_auc_score, 
                            precision_recall_curve, average_precision_score,
                            matthews_corrcoef, cohen_kappa_score)
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Tuple, List
import sys
import os
import pickle

# Add your models directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from anomaly_detector import AnomalyDetectionEngine  # Import your engine


class AnomalyDetectionEvaluator:
    """Comprehensive evaluation framework for anomaly detection"""
    
    def __init__(self, engine: AnomalyDetectionEngine, feature_names: List[str], 
                 win_size: int = 100, scaler: RobustScaler = None):
        self.engine = engine
        self.feature_names = feature_names
        self.win_size = win_size
        self.logger = logging.getLogger(__name__)
        self.predictions = []
        self.scores = []
        self.ground_truth = []
        self.timestamps = []
        self.scaler = scaler
    
    def prepare_data(self, csv_path: str) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """
        Prepare data from CSV for evaluation
        
        Args:
            csv_path: Path to test_data.csv
            
        Returns:
            Tuple of (data_windows, labels, timestamps)
        """
        if self.scaler is None:
            raise ValueError("RobustScaler must be provided before preparing data")
        
        df = pd.read_csv(csv_path)
        
        # Extract features (exclude date and is_anomaly)
        feature_cols = [col for col in df.columns if col not in ['date', 'is_anomaly']]
        
        # Normalize features using the fitted scaler
        data = df[feature_cols].values
        data = self.scaler.transform(data)
        
        # Create sliding windows
        data_windows = []
        labels = []
        timestamps = []
        
        for i in range(len(data) - self.win_size + 1):
            window = data[i:i + self.win_size]
            # Use the label of the last timestep in the window
            label = df['is_anomaly'].iloc[i + self.win_size - 1]
            timestamp = df['date'].iloc[i + self.win_size - 1]
            
            data_windows.append(window)
            labels.append(label)
            timestamps.append(timestamp)
        
        self.logger.info(f"Prepared {len(data_windows)} windows from {len(df)} records")
        return data_windows, np.array(labels), timestamps
        

    
    def evaluate(self, csv_path: str, plot_dir: str = './evaluation_plots', 
                apply_adjustment: bool = False, apply_lag: bool = False, 
                lag_tolerance: int = 1) -> Dict:
        """
        Run full evaluation on test set
        
        Args:
            csv_path: Path to test_data.csv
            plot_dir: Directory to save evaluation plots
            apply_adjustment: Whether to apply segment adjustment strategy
            apply_lag: Whether to apply lag tolerance adjustment
            lag_tolerance: Lag tolerance in timesteps (default: 1)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        os.makedirs(plot_dir, exist_ok=True)
        
        # Prepare data
        data_windows, ground_truth, timestamps = self.prepare_data(csv_path)
        self.ground_truth = ground_truth
        self.timestamps = timestamps
        
        # Get predictions
        self.logger.info("Running inference on test set...")
        for i, window in enumerate(data_windows):
            try:
                score, feature_contrib = self.engine.detect_anomaly(window)
                self.scores.append(score)
                prediction = self.engine.is_anomaly(score)
                self.predictions.append(1 if prediction else 0)
            except Exception as e:
                self.logger.error(f"Error processing window {i}: {e}")
                self.scores.append(0.0)
                self.predictions.append(0)
        
        self.predictions = np.array(self.predictions)
        self.scores = np.array(self.scores)
        
        # Calculate metrics with optional adjustments
        metrics = self._calculate_metrics(apply_adjustment=apply_adjustment, 
                                         apply_lag=apply_lag, 
                                         lag_tolerance=lag_tolerance)
        
        # Generate plots
        self._plot_confusion_matrix(plot_dir)
        self._plot_roc_curve(plot_dir)
        self._plot_pr_curve(plot_dir)
        self._plot_score_distribution(plot_dir)
        self._plot_predictions_timeline(plot_dir)
        self._plot_feature_importance(plot_dir, data_windows)
        
        # Save results
        self._save_results(metrics, plot_dir, apply_adjustment, apply_lag, lag_tolerance)
        
        return metrics
    
    def _apply_adjustment_strategy(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
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
    
    def _apply_lag_tolerance(self, gt: np.ndarray, pred: np.ndarray, lag_tolerance: int = 1) -> np.ndarray:
        """
        Apply lag tolerance: Convert predictions that occur within lag_tolerance timesteps
        AFTER a ground truth anomaly into correct detections (to account for detection delay).
        
        This handles:
        - Lagged detections after anomaly ends (e.g., GT=[3,4], Pred=5 with lag=1 → TP)
        - Detections within anomalous segments (already TP)
        
        Args:
            gt: ground truth labels
            pred: predictions
            lag_tolerance: number of timesteps after anomaly to allow detection (default: 1)
        
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
        
        # For each GT segment, check predictions within lag tolerance window AFTER segment
        for seg_start, seg_end in gt_segments:
            # Check if there's a detection in the segment itself
            detection_in_segment = np.any(pred[seg_start:seg_end + 1] == 1)
            
            # Also check for lagged detections after the segment ends
            lag_window_start = seg_end + 1
            lag_window_end = min(len(gt) - 1, seg_end + lag_tolerance)
            
            detection_in_lag_window = False
            if lag_window_start <= lag_window_end:
                detection_in_lag_window = np.any(pred[lag_window_start:lag_window_end + 1] == 1)
            
            # If detection found either in segment or in lag window after segment
            if detection_in_segment or detection_in_lag_window:
                # Mark entire GT segment as detected (TP)
                for j in range(seg_start, seg_end + 1):
                    adjusted_pred[j] = 1
                
                # Also mark lagged detections as TP (convert FP to TP)
                if lag_window_start <= lag_window_end:
                    for j in range(lag_window_start, lag_window_end + 1):
                        if pred[j] == 1:
                            adjusted_pred[j] = 1
        
        return adjusted_pred
    
    def _calculate_metrics(self, apply_adjustment: bool = False, apply_lag: bool = False, 
                          lag_tolerance: int = 1) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            apply_adjustment: Whether to apply segment adjustment strategy
            apply_lag: Whether to apply lag tolerance adjustment
            lag_tolerance: Lag tolerance in timesteps
        """
        predictions_to_use = self.predictions.copy()
        
        # Apply adjustments if requested
        if apply_adjustment:
            self.logger.info("Applying adjustment strategy...")
            predictions_to_use = self._apply_adjustment_strategy(self.ground_truth, predictions_to_use)
        
        if apply_lag:
            self.logger.info(f"Applying lag tolerance adjustment (tolerance={lag_tolerance})...")
            predictions_to_use = self._apply_lag_tolerance(self.ground_truth, predictions_to_use, lag_tolerance)
        
        # Find indices of FP and FN
        fp_indices = np.where((predictions_to_use == 1) & (self.ground_truth == 0))[0]
        fn_indices = np.where((predictions_to_use == 0) & (self.ground_truth == 1))[0]
        
        # Get anomaly scores for FP and FN
        fp_scores = self.scores[fp_indices]
        fn_scores = self.scores[fn_indices]
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(self.ground_truth, predictions_to_use).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = precision_score(self.ground_truth, self.predictions, zero_division=0)
        recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Advanced metrics
        mcc = matthews_corrcoef(self.ground_truth, self.predictions)
        
        # ROC-AUC
        roc_auc = roc_auc_score(self.ground_truth, self.scores) if len(np.unique(self.ground_truth)) > 1 else 0
        
        # PR-AUC
        pr_auc = average_precision_score(self.ground_truth, self.scores) if len(np.unique(self.ground_truth)) > 1 else 0
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(self.ground_truth, self.predictions)
        
        # Anomaly-specific metrics
        n_anomalies_true = np.sum(self.ground_truth)
        n_anomalies_pred = np.sum(self.predictions)
        detection_rate = tp / n_anomalies_true if n_anomalies_true > 0 else 0
        false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'mcc': mcc,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'kappa': kappa,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'threshold': self.engine.threshold,
            'confusion_matrix': {
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            },
            'dataset_stats': {
                'total_samples': len(self.ground_truth),
                'true_anomalies': int(n_anomalies_true),
                'predicted_anomalies': int(n_anomalies_pred)
            },
            'fp_indices': fp_indices.tolist(),
            'fn_indices': fn_indices.tolist(),
            'fp_scores': fp_scores.tolist(),
            'fn_scores': fn_scores.tolist()
        }
        
        return metrics
    
    def _plot_confusion_matrix(self, plot_dir: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.ground_truth, self.predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def _plot_roc_curve(self, plot_dir: str):
        """Plot ROC curve"""
        if len(np.unique(self.ground_truth)) < 2:
            self.logger.warning("Cannot plot ROC curve: only one class present")
            return
            
        fpr, tpr, _ = roc_curve(self.ground_truth, self.scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'roc_curve.png'), dpi=300)
        plt.close()
    
    def _plot_pr_curve(self, plot_dir: str):
        """Plot Precision-Recall curve"""
        if len(np.unique(self.ground_truth)) < 2:
            self.logger.warning("Cannot plot PR curve: only one class present")
            return
            
        precision, recall, _ = precision_recall_curve(self.ground_truth, self.scores)
        pr_auc = average_precision_score(self.ground_truth, self.scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'pr_curve.png'), dpi=300)
        plt.close()
    
    def _plot_score_distribution(self, plot_dir: str):
        """Plot distribution of anomaly scores"""
        plt.figure(figsize=(10, 6))
        normal_scores = self.scores[self.ground_truth == 0]
        anomaly_scores = self.scores[self.ground_truth == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        if len(anomaly_scores) > 0:
            plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
        
        plt.axvline(self.engine.threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold ({self.engine.threshold:.4f})')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'score_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_predictions_timeline(self, plot_dir: str):
        """Plot predictions over time"""
        plt.figure(figsize=(14, 6))
        x = np.arange(len(self.scores))
        
        plt.plot(x, self.scores, label='Anomaly Scores', alpha=0.7)
        plt.axhline(self.engine.threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.engine.threshold:.4f})')
        
        # Highlight correct and incorrect predictions
        correct = self.predictions == self.ground_truth
        incorrect = self.predictions != self.ground_truth
        
        plt.scatter(x[correct], self.scores[correct], alpha=0.3, s=20, 
                   label='Correct Predictions', color='green')
        plt.scatter(x[incorrect], self.scores[incorrect], alpha=0.5, s=50, 
                   label='Incorrect Predictions', color='orange', marker='x')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores Timeline')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'predictions_timeline.png'), dpi=300)
        plt.close()
    
    def _plot_feature_importance(self, plot_dir: str, data_windows: List[np.ndarray]):
        """Analyze feature importance in anomalies"""
        feature_contributions_list = []
        
        for i, window in enumerate(data_windows):
            try:
                _, feature_contrib = self.engine.detect_anomaly(window)
                if self.predictions[i] == 1:  # Only for predicted anomalies
                    feature_contributions_list.append(feature_contrib)
            except Exception as e:
                self.logger.error(f"Error extracting features: {e}")
        
        if feature_contributions_list:
            avg_feature_contrib = np.mean(feature_contributions_list, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.bar(self.feature_names, avg_feature_contrib)
            plt.xlabel('Features')
            plt.ylabel('Average Contribution')
            plt.title('Feature Importance in Detected Anomalies')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'feature_importance.png'), dpi=300)
            plt.close()
    
    def _save_results(self, metrics: Dict, plot_dir: str, apply_adjustment: bool = False, 
                     apply_lag: bool = False, lag_tolerance: int = 1):
        """Save evaluation results to file"""
        results_path = os.path.join(plot_dir, 'evaluation_results.txt')
        
        with open(results_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ANOMALY DETECTION EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Threshold: {metrics['threshold']:.4f}\n\n")
            
            # Write adjustment strategy info
            f.write("ADJUSTMENT STRATEGIES APPLIED:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Segment Adjustment: {'Yes' if apply_adjustment else 'No'}\n")
            f.write(f"Lag Tolerance: {'Yes (tolerance={} timesteps)'.format(lag_tolerance) if apply_lag else 'No'}\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 60 + "\n")
            for key, value in metrics['dataset_stats'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nCONFUSION MATRIX:\n")
            f.write("-" * 60 + "\n")
            for key, value in metrics['confusion_matrix'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nPERFORMANCE METRICS:\n")
            f.write("-" * 60 + "\n")
            metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                          'mcc', 'roc_auc', 'pr_auc', 'kappa', 'detection_rate', 
                          'false_positive_rate']
            for metric in metric_names:
                if metric in metrics:
                    f.write(f"{metric}: {metrics[metric]:.4f}\n")
            
            # Write FP and FN indices
            f.write("\nFALSE POSITIVES (FP):\n")
            f.write("-" * 60 + "\n")
            fp_indices = metrics['fp_indices']
            fp_scores = metrics['fp_scores']
            if len(fp_indices) > 0:
                f.write(f"Count: {len(fp_indices)}\n")
                f.write(f"Indices: {fp_indices}\n\n")
                f.write("Details (Index | Anomaly Score | Timestamp):\n")
                for i, idx in enumerate(fp_indices):
                    score = fp_scores[i]
                    if self.timestamps is not None and idx < len(self.timestamps):
                        timestamp = self.timestamps[idx]
                        f.write(f"  {idx:6d} | {score:12.6f} | {timestamp}\n")
                    else:
                        f.write(f"  {idx:6d} | {score:12.6f}\n")
            else:
                f.write("Count: 0\nNo false positives detected.\n")
            
            f.write("\nFALSE NEGATIVES (FN):\n")
            f.write("-" * 60 + "\n")
            fn_indices = metrics['fn_indices']
            fn_scores = metrics['fn_scores']
            if len(fn_indices) > 0:
                f.write(f"Count: {len(fn_indices)}\n")
                f.write(f"Indices: {fn_indices}\n\n")
                f.write("Details (Index | Anomaly Score | Timestamp):\n")
                for i, idx in enumerate(fn_indices):
                    score = fn_scores[i]
                    if self.timestamps is not None and idx < len(self.timestamps):
                        timestamp = self.timestamps[idx]
                        f.write(f"  {idx:6d} | {score:12.6f} | {timestamp}\n")
                    else:
                        f.write(f"  {idx:6d} | {score:12.6f}\n")
            else:
                f.write("Count: 0\nNo false negatives detected.\n")
        
        self.logger.info(f"Results saved to {results_path}")


def train_scaler(dataset_csv_path: str, feature_names: List[str]) -> RobustScaler:
    """
    Train RobustScaler on the full dataset
    
    Args:
        dataset_csv_path: Path to dataset.csv (training data)
        feature_names: List of feature column names
        
    Returns:
        Fitted RobustScaler instance
    """
    df = pd.read_csv(dataset_csv_path)
    
    # Extract features
    feature_cols = [col for col in df.columns if col in feature_names]
    data = df[feature_cols].values
    
    # Train and fit scaler
    scaler = RobustScaler()
    scaler.fit(data)
    
    logging.info(f"RobustScaler trained on {len(data)} samples from {dataset_csv_path}")
    return scaler


def save_scaler(scaler: RobustScaler, path: str):
    """Save scaler for later use"""
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {path}")


def load_scaler(path: str) -> RobustScaler:
    """Load previously saved scaler"""
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    logging.info(f"Scaler loaded from {path}")
    return scaler


import argparse


# Usage Example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Anomaly Detection Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.csv for scaler training')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test_data.csv for evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (e.g., all_checkpoint.pth)')
    parser.add_argument('--threshold', type=float, required=True, help='Anomaly detection threshold')
    parser.add_argument('--features', type=str, default='tt,rh,pp,ws,wd,sr,rr', 
                       help='Comma-separated feature names')
    parser.add_argument('--window-size', type=int, default=100, help='Sliding window size')
    parser.add_argument('--output-dir', type=str, default='./evaluation_plots', help='Output directory for plots and results')
    parser.add_argument('--save-scaler', type=str, default=None, help='Path to save the scaler (optional)')
    parser.add_argument('--load-scaler', type=str, default=None, help='Path to load a previously saved scaler (optional)')
    parser.add_argument('--model-config', type=str, default='{}', help='JSON string of model config')
    parser.add_argument('--apply-adjustment', action='store_true', help='Apply segment adjustment strategy')
    parser.add_argument('--apply-lag', action='store_true', help='Apply lag tolerance adjustment')
    parser.add_argument('--lag-tolerance', type=int, default=1, help='Lag tolerance in timesteps (default: 1)')
    
    args = parser.parse_args()
    
    # Parse features
    feature_names = [f.strip() for f in args.features.split(',')]
    
    # Parse model config
    import json
    try:
        model_config = json.loads(args.model_config)
    except json.JSONDecodeError:
        model_config = {}
    
    # Step 1: Load or train RobustScaler
    if args.load_scaler:
        scaler = load_scaler(args.load_scaler)
    else:
        scaler = train_scaler(args.dataset, feature_names)
        if args.save_scaler:
            save_scaler(scaler, args.save_scaler)
    
    # Step 2: Create engine and evaluator
    engine = AnomalyDetectionEngine(
        model_config=model_config,
        checkpoint_path=args.checkpoint,
        feature_names=feature_names,
        win_size=args.window_size,
        use_fixed_threshold=True,
        fixed_threshold=args.threshold
    )
    
    evaluator = AnomalyDetectionEvaluator(engine, feature_names, win_size=args.window_size, scaler=scaler)
    
    # Step 3: Run evaluation on test set
    metrics = evaluator.evaluate(args.test_data, plot_dir=args.output_dir,
                                apply_adjustment=args.apply_adjustment,
                                apply_lag=args.apply_lag,
                                lag_tolerance=args.lag_tolerance)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Detection Rate: {metrics['detection_rate']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print("=" * 60)
    
    if args.apply_adjustment or args.apply_lag:
        print("\nAdjustments Applied:")
        if args.apply_adjustment:
            print("  - Segment Adjustment Strategy: Enabled")
        if args.apply_lag:
            print(f"  - Lag Tolerance: Enabled (tolerance={args.lag_tolerance} timesteps)")