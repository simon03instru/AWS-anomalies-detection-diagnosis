import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                            f1_score, roc_auc_score, accuracy_score)
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import sys
import os
import pickle
import json
import argparse

# Add your models directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from anomaly_detector_test import AnomalyDetectionEngine


class MultiThresholdEvaluator:
    """Evaluate anomaly detection across multiple thresholds"""
    
    def __init__(self, engine_base_config: Dict, feature_names: List[str], 
                 win_size: int = 100, scaler: RobustScaler = None):
        """
        Initialize multi-threshold evaluator
        
        Args:
            engine_base_config: Base configuration for the engine (without threshold)
            feature_names: List of feature names
            win_size: Window size for sliding windows
            scaler: Fitted RobustScaler instance
        """
        self.engine_base_config = engine_base_config
        self.feature_names = feature_names
        self.win_size = win_size
        self.scaler = scaler
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.all_scores = None
        self.ground_truth = None
        self.timestamps = None
        self.data_windows = None
        
    def prepare_data(self, csv_path: str) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """Prepare data from CSV for evaluation"""
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
            label = df['is_anomaly'].iloc[i + self.win_size - 1]
            timestamp = df['date'].iloc[i + self.win_size - 1]
            
            data_windows.append(window)
            labels.append(label)
            timestamps.append(timestamp)
        
        self.logger.info(f"Prepared {len(data_windows)} windows from {len(df)} records")
        return data_windows, np.array(labels), timestamps
    
    def compute_anomaly_scores(self, csv_path: str):
        """
        Compute anomaly scores for all windows (done once)
        
        Args:
            csv_path: Path to test_data.csv
        """
        # Prepare data
        self.data_windows, self.ground_truth, self.timestamps = self.prepare_data(csv_path)
        
        # Create a temporary engine just to get scores
        temp_engine = AnomalyDetectionEngine(
            model_config=self.engine_base_config.get('model_config', {}),
            checkpoint_path=self.engine_base_config['checkpoint_path'],
            feature_names=self.feature_names,
            win_size=self.win_size,
            use_fixed_threshold=True,
            fixed_threshold=0.5  # Arbitrary, we'll override later
        )
        
        # Get scores for all windows
        self.logger.info("Computing anomaly scores for all windows...")
        scores = []
        for i, window in enumerate(self.data_windows):
            try:
                score, _ = temp_engine.detect_anomaly(window)
                scores.append(score)
            except Exception as e:
                self.logger.error(f"Error processing window {i}: {e}")
                scores.append(0.0)
        
        self.all_scores = np.array(scores)
        self.logger.info(f"Computed {len(self.all_scores)} anomaly scores")
    
    def evaluate_threshold(self, threshold: float, apply_adjustment: bool = False,
                          apply_lag: bool = False, lag_tolerance: int = 1) -> Dict:
        """
        Evaluate metrics for a single threshold
        
        Args:
            threshold: Threshold value to test
            apply_adjustment: Whether to apply segment adjustment
            apply_lag: Whether to apply lag tolerance
            lag_tolerance: Lag tolerance in timesteps
            
        Returns:
            Dictionary with metrics for this threshold
        """
        # Generate predictions based on threshold
        predictions = (self.all_scores >= threshold).astype(int)
        ground_truth_to_use = self.ground_truth.copy()
        predictions_to_use = predictions.copy()
        
        # Apply adjustments if requested
        if apply_lag:
            ground_truth_to_use = self._apply_lag_tolerance(
                ground_truth_to_use, predictions_to_use, lag_tolerance
            )
        
        if apply_adjustment:
            predictions_to_use = self._apply_adjustment_strategy(
                ground_truth_to_use, predictions_to_use
            )
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(ground_truth_to_use, predictions_to_use).ravel()
        
        accuracy = accuracy_score(ground_truth_to_use, predictions_to_use)
        precision = precision_score(ground_truth_to_use, predictions_to_use, zero_division=0)
        recall = recall_score(ground_truth_to_use, predictions_to_use, zero_division=0)
        f1 = f1_score(ground_truth_to_use, predictions_to_use, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Detection rate
        n_anomalies_true = np.sum(ground_truth_to_use)
        detection_rate = tp / n_anomalies_true if n_anomalies_true > 0 else 0
        
        # False positive rate
        false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_predictions': int(np.sum(predictions_to_use)),
            'n_true_anomalies': int(n_anomalies_true)
        }
    
    def _apply_lag_tolerance(self, gt: np.ndarray, pred: np.ndarray, 
                            lag_tolerance: int = 1) -> np.ndarray:
        """Apply lag tolerance adjustment"""
        adjusted_gt = gt.copy()
        
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
        
        # Extend acceptable detection window
        for seg_start, seg_end in gt_segments:
            lag_window_start = seg_end + 1
            lag_window_end = min(len(gt) - 1, seg_end + lag_tolerance)
            
            if lag_window_start <= lag_window_end:
                for j in range(lag_window_start, lag_window_end + 1):
                    if pred[j] == 1 and gt[j] == 0:
                        adjusted_gt[j] = 1
        
        return adjusted_gt
    
    def _apply_adjustment_strategy(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Apply segment adjustment strategy"""
        adjusted_pred = pred.copy()
        anomaly_state = False
        
        for i in range(len(gt)):
            if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                
                # Backward fill
                for j in range(i, -1, -1):
                    if gt[j] == 0:
                        break
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
                
                # Forward fill
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
            
            elif gt[i] == 0:
                anomaly_state = False
            
            if anomaly_state:
                adjusted_pred[i] = 1
        
        return adjusted_pred
    
    def evaluate_multiple_thresholds(self, thresholds: List[float], 
                                    csv_path: str = None,
                                    apply_adjustment: bool = False,
                                    apply_lag: bool = False,
                                    lag_tolerance: int = 1) -> pd.DataFrame:
        """
        Evaluate multiple thresholds
        
        Args:
            thresholds: List of threshold values to test
            csv_path: Path to test data (if scores not already computed)
            apply_adjustment: Whether to apply segment adjustment
            apply_lag: Whether to apply lag tolerance
            lag_tolerance: Lag tolerance in timesteps
            
        Returns:
            DataFrame with results for all thresholds
        """
        # Compute scores if not already done
        if self.all_scores is None:
            if csv_path is None:
                raise ValueError("csv_path must be provided if scores not computed")
            self.compute_anomaly_scores(csv_path)
        
        # Evaluate each threshold
        results = []
        self.logger.info(f"Evaluating {len(thresholds)} thresholds...")
        
        for i, threshold in enumerate(thresholds):
            self.logger.info(f"Evaluating threshold {i+1}/{len(thresholds)}: {threshold:.4f}")
            metrics = self.evaluate_threshold(threshold, apply_adjustment, 
                                             apply_lag, lag_tolerance)
            results.append(metrics)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        return df_results
    
    def plot_threshold_analysis(self, df_results: pd.DataFrame, output_dir: str):
        """
        Generate comprehensive plots for threshold analysis
        
        Args:
            df_results: DataFrame with results from evaluate_multiple_thresholds
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Main metrics vs threshold
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # F1-Score
        axes[0, 0].plot(df_results['threshold'], df_results['f1_score'], 
                       marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].set_title('F1-Score vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        best_f1_idx = df_results['f1_score'].idxmax()
        axes[0, 0].axvline(df_results.loc[best_f1_idx, 'threshold'], 
                          color='red', linestyle='--', alpha=0.5,
                          label=f"Best: {df_results.loc[best_f1_idx, 'threshold']:.4f}")
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(df_results['threshold'], df_results['accuracy'], 
                       marker='o', linewidth=2, markersize=4, color='green')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 0].plot(df_results['threshold'], df_results['precision'], 
                       marker='o', linewidth=2, markersize=4, label='Precision')
        axes[1, 0].plot(df_results['threshold'], df_results['recall'], 
                       marker='s', linewidth=2, markersize=4, label='Recall')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Detection Rate
        axes[1, 1].plot(df_results['threshold'], df_results['detection_rate'], 
                       marker='o', linewidth=2, markersize=4, color='purple')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Detection Rate')
        axes[1, 1].set_title('Detection Rate vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_metrics.png'), dpi=300)
        plt.close()
        
        # 2. Precision-Recall Tradeoff
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['recall'], df_results['precision'], 
                marker='o', linewidth=2, markersize=6)
        
        # Annotate some points
        step = max(1, len(df_results) // 10)
        for i in range(0, len(df_results), step):
            plt.annotate(f"{df_results.loc[i, 'threshold']:.3f}",
                        (df_results.loc[i, 'recall'], df_results.loc[i, 'precision']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Tradeoff')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_tradeoff.png'), dpi=300)
        plt.close()
        
        # 3. Confusion Matrix Components
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(df_results['threshold'], df_results['tp'], 
                       marker='o', linewidth=2, color='green', label='TP')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('True Positives vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df_results['threshold'], df_results['fp'], 
                       marker='o', linewidth=2, color='red', label='FP')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('False Positives vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(df_results['threshold'], df_results['tn'], 
                       marker='o', linewidth=2, color='blue', label='TN')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('True Negatives vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df_results['threshold'], df_results['fn'], 
                       marker='o', linewidth=2, color='orange', label='FN')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('False Negatives vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_components.png'), dpi=300)
        plt.close()
        
        self.logger.info(f"Plots saved to {output_dir}")
    
    def save_results(self, df_results: pd.DataFrame, output_dir: str, 
                    apply_adjustment: bool = False, apply_lag: bool = False,
                    lag_tolerance: int = 1):
        """Save results to CSV and text summary"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results to CSV
        csv_path = os.path.join(output_dir, 'threshold_results.csv')
        df_results.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")
        
        # Save summary to text file
        txt_path = os.path.join(output_dir, 'threshold_summary.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-THRESHOLD EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Thresholds Tested: {len(df_results)}\n")
            f.write(f"Threshold Range: [{df_results['threshold'].min():.4f}, "
                   f"{df_results['threshold'].max():.4f}]\n\n")
            
            f.write("ADJUSTMENT STRATEGIES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Segment Adjustment: {'Yes' if apply_adjustment else 'No'}\n")
            f.write(f"Lag Tolerance: {'Yes (tolerance={} timesteps)'.format(lag_tolerance) if apply_lag else 'No'}\n\n")
            
            # Best thresholds for each metric
            f.write("OPTIMAL THRESHOLDS:\n")
            f.write("-" * 80 + "\n")
            
            metrics = ['f1_score', 'accuracy', 'precision', 'recall', 'detection_rate']
            for metric in metrics:
                best_idx = df_results[metric].idxmax()
                best_row = df_results.loc[best_idx]
                f.write(f"\nBest {metric.upper()}:\n")
                f.write(f"  Threshold: {best_row['threshold']:.4f}\n")
                f.write(f"  {metric}: {best_row[metric]:.4f}\n")
                f.write(f"  Accuracy: {best_row['accuracy']:.4f}\n")
                f.write(f"  Precision: {best_row['precision']:.4f}\n")
                f.write(f"  Recall: {best_row['recall']:.4f}\n")
                f.write(f"  F1-Score: {best_row['f1_score']:.4f}\n")
                f.write(f"  Detection Rate: {best_row['detection_rate']:.4f}\n")
            
            # Top 5 by F1-Score
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 5 THRESHOLDS BY F1-SCORE:\n")
            f.write("=" * 80 + "\n\n")
            
            top5 = df_results.nlargest(5, 'f1_score')
            for idx, row in top5.iterrows():
                f.write(f"Threshold: {row['threshold']:.4f}\n")
                f.write(f"  F1-Score: {row['f1_score']:.4f}\n")
                f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
                f.write(f"  Precision: {row['precision']:.4f}\n")
                f.write(f"  Recall: {row['recall']:.4f}\n")
                f.write(f"  Detection Rate: {row['detection_rate']:.4f}\n")
                f.write(f"  TP/FP/TN/FN: {row['tp']}/{row['fp']}/{row['tn']}/{row['fn']}\n\n")
        
        self.logger.info(f"Summary saved to {txt_path}")


def train_scaler(dataset_csv_path: str, feature_names: List[str]) -> RobustScaler:
    """Train RobustScaler on the full dataset"""
    df = pd.read_csv(dataset_csv_path)
    feature_cols = [col for col in df.columns if col in feature_names]
    data = df[feature_cols].values
    
    scaler = RobustScaler()
    scaler.fit(data)
    
    logging.info(f"RobustScaler trained on {len(data)} samples")
    return scaler


def load_scaler(path: str) -> RobustScaler:
    """Load previously saved scaler"""
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    logging.info(f"Scaler loaded from {path}")
    return scaler


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-Threshold Anomaly Detection Evaluation')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to dataset.csv for scaler training')
    parser.add_argument('--test-data', type=str, required=True, 
                       help='Path to test_data.csv for evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--features', type=str, default='tt,rh,pp,ws,wd,sr,rr', 
                       help='Comma-separated feature names')
    parser.add_argument('--window-size', type=int, default=100, 
                       help='Sliding window size')
    parser.add_argument('--output-dir', type=str, default='./threshold_evaluation', 
                       help='Output directory')
    parser.add_argument('--load-scaler', type=str, default=None, 
                       help='Path to load saved scaler')
    parser.add_argument('--model-config', type=str, default='{}', 
                       help='JSON string of model config')
    
    # Threshold configuration
    parser.add_argument('--threshold-min', type=float, default=0.1, 
                       help='Minimum threshold to test')
    parser.add_argument('--threshold-max', type=float, default=0.9, 
                       help='Maximum threshold to test')
    parser.add_argument('--threshold-step', type=float, default=0.05, 
                       help='Step size between thresholds')
    parser.add_argument('--threshold', type=float, nargs='+', default=None,
                       help='Discrete thresholds to test (e.g., --threshold 0.5 0.6 0.7)')
    parser.add_argument('--thresholds', type=str, default=None, 
                       help='Comma-separated list of specific thresholds (alternative to --threshold)')
    
    # Adjustment strategies
    parser.add_argument('--apply-adjustment', action='store_true', 
                       help='Apply segment adjustment strategy')
    parser.add_argument('--apply-lag', action='store_true', 
                       help='Apply lag tolerance adjustment')
    parser.add_argument('--lag-tolerance', type=int, default=1, 
                       help='Lag tolerance in timesteps')
    
    args = parser.parse_args()
    
    # Parse features
    feature_names = [f.strip() for f in args.features.split(',')]
    
    # Parse model config
    try:
        model_config = json.loads(args.model_config)
    except json.JSONDecodeError:
        model_config = {}
    
    # Load or train scaler
    if args.load_scaler:
        scaler = load_scaler(args.load_scaler)
    else:
        scaler = train_scaler(args.dataset, feature_names)
    
    # Generate threshold list
    if args.threshold is not None:
        # Use discrete thresholds from --threshold argument
        thresholds = args.threshold
    elif args.thresholds:
        # Use comma-separated thresholds from --thresholds argument
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    else:
        # Use range-based thresholds
        thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, 
                              args.threshold_step).tolist()
    
    print(f"\nTesting {len(thresholds)} thresholds: {thresholds}")
    
    # Create engine base config
    engine_base_config = {
        'model_config': model_config,
        'checkpoint_path': args.checkpoint
    }
    
    # Create evaluator
    evaluator = MultiThresholdEvaluator(
        engine_base_config=engine_base_config,
        feature_names=feature_names,
        win_size=args.window_size,
        scaler=scaler
    )
    
    # Run evaluation
    df_results = evaluator.evaluate_multiple_thresholds(
        thresholds=thresholds,
        csv_path=args.test_data,
        apply_adjustment=args.apply_adjustment,
        apply_lag=args.apply_lag,
        lag_tolerance=args.lag_tolerance
    )
    
    # Generate plots
    evaluator.plot_threshold_analysis(df_results, args.output_dir)
    
    # Save results
    evaluator.save_results(df_results, args.output_dir, 
                          args.apply_adjustment, args.apply_lag, args.lag_tolerance)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    best_f1_idx = df_results['f1_score'].idxmax()
    best_f1_row = df_results.loc[best_f1_idx]
    
    print(f"\nBest F1-Score Configuration:")
    print(f"  Threshold: {best_f1_row['threshold']:.4f}")
    print(f"  F1-Score: {best_f1_row['f1_score']:.4f}")
    print(f"  Accuracy: {best_f1_row['accuracy']:.4f}")
    print(f"  Precision: {best_f1_row['precision']:.4f}")
    print(f"  Recall: {best_f1_row['recall']:.4f}")
    print(f"  Detection Rate: {best_f1_row['detection_rate']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 80)