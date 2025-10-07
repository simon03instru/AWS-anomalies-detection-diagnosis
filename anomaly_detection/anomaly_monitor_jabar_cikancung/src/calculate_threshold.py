#!/usr/bin/env python3
"""
Threshold calculation script for anomaly detection model
Calculates anomaly scores for last 5000 rows and determines 95th percentile threshold
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add models directory to path (adjust path as needed)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

# Import your existing components
try:
    from model.anomaly_main import Trainer
    from scaler_manager import WeatherDataScaler
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure anomaly_main.py and scaler_manager.py are accessible")
    sys.exit(1)

class ThresholdCalculator:
    """Calculate optimal threshold for anomaly detection model"""
    
    def __init__(self, model_config: Dict, checkpoint_path: str, 
                 dataset_path: str, feature_names: List[str], 
                 win_size: int = 100, temperature: int = 50):
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.feature_names = feature_names
        self.win_size = win_size
        self.temperature = temperature
        
        self.model = None
        self.scaler = None
        
        self._load_model()
        self._load_scaler()
    
    def _load_model(self):
        """Load the trained anomaly detection model"""
        try:
            trainer = Trainer(self.model_config)
            trainer.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
            )
            trainer.model.eval()
            self.model = trainer
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_scaler(self):
        """Load the scaler if available"""
        try:
            self.scaler = WeatherDataScaler()
            logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}")
            self.scaler = None
    
    def _kl_loss(self, p, q):
        """KL divergence loss"""
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)
    
    def detect_anomaly_single_window(self, data_window: np.ndarray) -> float:
        """
        Detect anomaly in a single data window
        
        Args:
            data_window: numpy array of shape (win_size, n_features)
            
        Returns:
            Anomaly score (float)
        """
        try:
            # Prepare input for model
            model_input = torch.FloatTensor(data_window).unsqueeze(0)
            
            with torch.no_grad():
                output, series, prior, _ = self.model.model(model_input)
                
                # Calculate reconstruction loss
                criterion = torch.nn.MSELoss(reduction='none')
                loss = torch.mean(criterion(model_input, output), dim=-1)
                loss_per_feature = criterion(model_input, output)
                
                # Calculate KL losses
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    series_loss_term = self._kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, self.win_size)).detach()) * self.temperature
                    prior_loss_term = self._kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, self.win_size)),
                        series[u].detach()) * self.temperature
                    
                    if u == 0:
                        series_loss = series_loss_term
                        prior_loss = prior_loss_term
                    else:
                        series_loss += series_loss_term
                        prior_loss += prior_loss_term
                
                # Convert to numpy
                loss_per_feature = loss_per_feature.detach().cpu().numpy()
                
                # Use the raw loss values from the latest timestep
                latest_timestep_loss = loss_per_feature[0, -1, :]
                
                # Anomaly score is the mean of raw losses at the latest timestep
                final_anomaly_score = float(np.mean(latest_timestep_loss))
                
                return final_anomaly_score
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return 0.0
    
    def load_and_prepare_data(self, num_rows: int = 5000):
        """
        Load and prepare data from dataset
        
        Args:
            num_rows: Number of rows to load from the end of dataset
            
        Returns:
            Prepared data array or None if error
        """
        try:
            # Load dataset
            if not os.path.exists(self.dataset_path):
                logger.error(f"Dataset not found: {self.dataset_path}")
                return None
            
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset: {len(df)} total rows")
            
            # Take last num_rows
            if len(df) < num_rows:
                logger.warning(f"Dataset has only {len(df)} rows, using all available")
                data_subset = df.copy()
            else:
                data_subset = df.tail(num_rows).copy()
            
            logger.info(f"Using last {len(data_subset)} rows from dataset")
            
            # Clean the data - handle missing values (-999.0)
            data_subset = data_subset.replace(-999.0, np.nan)
            data_subset = data_subset.fillna(method='ffill').fillna(method='bfill')
            
            # Check if all required features are present
            missing_features = [f for f in self.feature_names if f not in data_subset.columns]
            if missing_features:
                logger.error(f"Missing features in dataset: {missing_features}")
                logger.info(f"Available columns: {list(data_subset.columns)}")
                return None
            
            # Extract feature data
            feature_data = data_subset[self.feature_names].values
            logger.info(f"Feature data shape: {feature_data.shape}")
            
            # Apply scaling if scaler available
            if self.scaler:
                try:
                    # Fit scaler if not already fitted
                    if not hasattr(self.scaler, 'is_fitted') or not self.scaler.is_fitted:
                        self.scaler.fit(feature_data)
                    
                    feature_data = self.scaler.scale_data(feature_data)
                    logger.info("Data scaled successfully")
                except Exception as e:
                    logger.warning(f"Scaling failed: {e}, using unscaled data")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            return None
    
    def calculate_all_scores(self, data: np.ndarray):
        """
        Calculate anomaly scores for all possible windows in the data
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            
        Returns:
            List of anomaly scores
        """
        anomaly_scores = []
        n_samples = len(data)
        
        if n_samples < self.win_size:
            logger.error(f"Not enough data: {n_samples} < {self.win_size}")
            return []
        
        # Calculate scores for all possible windows
        num_windows = n_samples - self.win_size + 1
        logger.info(f"Calculating anomaly scores for {num_windows} windows...")
        
        for i in range(num_windows):
            if i % 500 == 0:
                logger.info(f"Processing window {i+1}/{num_windows}")
            
            window = data[i:i+self.win_size]
            score = self.detect_anomaly_single_window(window)
            anomaly_scores.append(score)
        
        logger.info(f"Calculated {len(anomaly_scores)} anomaly scores")
        return anomaly_scores
    
    def calculate_threshold_and_stats(self, scores: List[float], percentile: float = 95.0):
        """
        Calculate threshold and statistics
        
        Args:
            scores: List of anomaly scores
            percentile: Percentile for threshold calculation
            
        Returns:
            Dictionary with threshold and statistics
        """
        if not scores:
            logger.error("No scores provided")
            return None
        
        scores_array = np.array(scores)
        
        # Calculate statistics
        stats = {
            'num_scores': len(scores),
            'mean_score': float(np.mean(scores_array)),
            'median_score': float(np.median(scores_array)),
            'std_score': float(np.std(scores_array)),
            'min_score': float(np.min(scores_array)),
            'max_score': float(np.max(scores_array)),
            'threshold_percentile': percentile,
            'threshold': float(np.percentile(scores_array, percentile)),
            'num_anomalies_detected': int(np.sum(scores_array > np.percentile(scores_array, percentile)))
        }
        
        # Calculate additional percentiles
        for p in [90, 95, 99]:
            stats[f'p{p}_score'] = float(np.percentile(scores_array, p))
        
        return stats
    
    def save_results(self, scores: List[float], stats: Dict, output_dir: str = "threshold_results"):
        """
        Save results to files
        
        Args:
            scores: List of anomaly scores
            stats: Statistics dictionary
            output_dir: Output directory
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save scores to CSV
            scores_df = pd.DataFrame({
                'window_index': range(len(scores)),
                'anomaly_score': scores,
                'is_anomaly': [s > stats['threshold'] for s in scores]
            })
            scores_csv_path = os.path.join(output_dir, 'anomaly_scores.csv')
            scores_df.to_csv(scores_csv_path, index=False)
            logger.info(f"Anomaly scores saved to: {scores_csv_path}")
            
            # Save statistics to JSON
            import json
            stats_json_path = os.path.join(output_dir, 'threshold_stats.json')
            with open(stats_json_path, 'w') as f:
                json.dump(stats, f, indent=4)
            logger.info(f"Statistics saved to: {stats_json_path}")
            
            # Create visualization
            self.create_visualization(scores, stats, output_dir)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def create_visualization(self, scores: List[float], stats: Dict, output_dir: str):
        """
        Create visualization plots
        
        Args:
            scores: List of anomaly scores
            stats: Statistics dictionary
            output_dir: Output directory
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Anomaly Detection Threshold Analysis', fontsize=16)
            
            # Plot 1: Score distribution
            axes[0, 0].hist(scores, bins=100, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(stats['threshold'], color='red', linestyle='--', 
                              label=f"95th Percentile: {stats['threshold']:.4f}")
            axes[0, 0].set_xlabel('Anomaly Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Anomaly Scores')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Score time series
            axes[0, 1].plot(scores, alpha=0.7, color='blue', linewidth=0.5)
            axes[0, 1].axhline(stats['threshold'], color='red', linestyle='--', 
                              label=f"Threshold: {stats['threshold']:.4f}")
            axes[0, 1].set_xlabel('Window Index')
            axes[0, 1].set_ylabel('Anomaly Score')
            axes[0, 1].set_title('Anomaly Scores Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Box plot
            axes[1, 0].boxplot(scores, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7))
            axes[1, 0].axhline(stats['threshold'], color='red', linestyle='--', 
                              label=f"Threshold: {stats['threshold']:.4f}")
            axes[1, 0].set_ylabel('Anomaly Score')
            axes[1, 0].set_title('Box Plot of Anomaly Scores')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Statistics summary (text)
            axes[1, 1].axis('off')
            stats_text = f"""
            Threshold Analysis Results
            ══════════════════════════
            
            Number of Windows: {stats['num_scores']:,}
            
            Score Statistics:
            • Mean: {stats['mean_score']:.6f}
            • Median: {stats['median_score']:.6f}
            • Std Dev: {stats['std_score']:.6f}
            • Min: {stats['min_score']:.6f}
            • Max: {stats['max_score']:.6f}
            
            Percentiles:
            • 90th: {stats['p90_score']:.6f}
            • 95th: {stats['p95_score']:.6f}
            • 99th: {stats['p99_score']:.6f}
            
            Threshold (95th percentile): {stats['threshold']:.6f}
            Anomalies Detected: {stats['num_anomalies_detected']:,}
            Anomaly Rate: {(stats['num_anomalies_detected']/stats['num_scores']*100):.2f}%
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=11, verticalalignment='top', fontfamily='monospace')
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'threshold_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {plot_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def run_analysis(self, dataset_path: str, num_rows: int = 5000, 
                     percentile: float = 95.0, output_dir: str = "threshold_results"):
        """
        Run complete threshold analysis
        
        Args:
            dataset_path: Path to dataset CSV file
            num_rows: Number of rows to analyze
            percentile: Percentile for threshold calculation
            output_dir: Output directory for results
        """
        logger.info("Starting threshold analysis...")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Analyzing last {num_rows} rows")
        logger.info(f"Threshold percentile: {percentile}%")
        
        # Update dataset path
        self.dataset_path = dataset_path
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data = self.load_and_prepare_data(num_rows)
        if data is None:
            logger.error("Failed to load data")
            return None
        
        # Calculate anomaly scores
        logger.info("Calculating anomaly scores...")
        scores = self.calculate_all_scores(data)
        if not scores:
            logger.error("Failed to calculate scores")
            return None
        
        # Calculate threshold and statistics
        logger.info("Calculating threshold and statistics...")
        stats = self.calculate_threshold_and_stats(scores, percentile)
        if stats is None:
            logger.error("Failed to calculate statistics")
            return None
        
        # Print results
        self.print_results(stats)
        
        # Save results
        logger.info("Saving results...")
        self.save_results(scores, stats, output_dir)
        
        logger.info("Analysis complete!")
        return stats
    
    def print_results(self, stats: Dict):
        """Print results to console"""
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS RESULTS")
        print("="*60)
        print(f"Number of windows analyzed: {stats['num_scores']:,}")
        print(f"Anomaly score statistics:")
        print(f"  • Mean: {stats['mean_score']:.6f}")
        print(f"  • Median: {stats['median_score']:.6f}")
        print(f"  • Standard deviation: {stats['std_score']:.6f}")
        print(f"  • Minimum: {stats['min_score']:.6f}")
        print(f"  • Maximum: {stats['max_score']:.6f}")
        print(f"\nPercentile scores:")
        print(f"  • 90th percentile: {stats['p90_score']:.6f}")
        print(f"  • 95th percentile: {stats['p95_score']:.6f}")
        print(f"  • 99th percentile: {stats['p99_score']:.6f}")
        print(f"\nRecommended threshold (95th percentile): {stats['threshold']:.6f}")
        print(f"Number of anomalies detected: {stats['num_anomalies_detected']:,}")
        print(f"Anomaly detection rate: {(stats['num_anomalies_detected']/stats['num_scores']*100):.2f}%")
        print("="*60)


def main():
    """Main function to run threshold calculation"""
    
    # Configuration - UPDATE THESE PATHS AND SETTINGS
    MODEL_CONFIG = {
        # Add your model configuration here
        # This should match the config used during training
    }
    
    CHECKPOINT_PATH = "../checkpoints/diy_pakem/all_checkpoint.pth"  # Update with correct path
    DATASET_PATH = "../dataset/dataset_fix/diy_pakem.csv"  # Update with correct dataset name

    FEATURE_NAMES = ['tt', 'rh', 'pp', 'ws', 'wd', 'sr', 'rr']  # Update with your features
    WIN_SIZE = 100
    NUM_ROWS = 5000
    PERCENTILE = 95.0
    OUTPUT_DIR = "threshold_results"
    
    # Validate paths
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Model checkpoint not found: {CHECKPOINT_PATH}")
        logger.info("Please update CHECKPOINT_PATH in the script")
        return
    
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found: {DATASET_PATH}")
        logger.info("Please update DATASET_PATH in the script")
        return
    
    try:
        # Create threshold calculator
        calculator = ThresholdCalculator(
            model_config=MODEL_CONFIG,
            checkpoint_path=CHECKPOINT_PATH,
            dataset_path=DATASET_PATH,
            feature_names=FEATURE_NAMES,
            win_size=WIN_SIZE
        )
        
        # Run analysis
        results = calculator.run_analysis(
            dataset_path=DATASET_PATH,
            num_rows=NUM_ROWS,
            percentile=PERCENTILE,
            output_dir=OUTPUT_DIR
        )
        
        if results:
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: {OUTPUT_DIR}/")
            print(f"✓ Recommended threshold: {results['threshold']:.6f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()