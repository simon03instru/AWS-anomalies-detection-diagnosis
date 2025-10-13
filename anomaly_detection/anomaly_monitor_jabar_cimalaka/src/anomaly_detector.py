import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os


# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

# Import your existing components
from anomaly_main import Trainer

class AnomalyDetectionEngine:
    """Core anomaly detection engine using the trained model"""
    
    def __init__(self, model_config: Dict, checkpoint_path: str, 
                 feature_names: List[str], win_size: int = 100,
                 use_fixed_threshold: bool = True, fixed_threshold: float = 0.1,
                 threshold_percentile: float = 95.0, temperature: int = 50):
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
        self.feature_names = feature_names
        self.win_size = win_size
        self.use_fixed_threshold = use_fixed_threshold
        self.fixed_threshold = fixed_threshold
        self.threshold_percentile = threshold_percentile
        self.temperature = temperature
        
        self.model = None
        self.threshold = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
        self._set_threshold()
        
    def _load_model(self):
        """Load the trained anomaly detection model"""
        try:
            trainer = Trainer(self.model_config)
            trainer.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
            )
            trainer.model.eval()
            self.model = trainer
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    def _set_threshold(self):
        """Set the anomaly threshold"""
        if self.use_fixed_threshold:
            self.threshold = self.fixed_threshold
            self.logger.info(f"Using FIXED threshold: {self.threshold}")
        else:
            try:
                if hasattr(self.model, 'test_loader') and self.model.test_loader:
                    _, test_energy = self.model.detect()
                    self.threshold = np.percentile(test_energy, self.threshold_percentile)
                    self.logger.info(f"Calculated dynamic threshold: {self.threshold:.4f}")
                else:
                    self.threshold = 0.5
                    self.logger.warning(f"Using default threshold: {self.threshold}")
            except Exception as e:
                self.logger.warning(f"Could not calculate dynamic threshold: {e}")
                self.threshold = self.fixed_threshold
                self.logger.info(f"Fallback to fixed threshold: {self.threshold}")
    
    def update_threshold(self, new_threshold: float):
        """Update the threshold value during runtime"""
        old_threshold = self.threshold
        self.threshold = new_threshold
        self.logger.info(f"Threshold updated from {old_threshold} to {new_threshold}")
        
    def detect_anomaly(self, data_window: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect anomaly in the data window
        
        Args:
            data_window: numpy array of shape (win_size, n_features)
            
        Returns:
            Tuple of (anomaly_score, feature_contributions)
        """
        criterion = nn.MSELoss(reduction='none')
        attens_energy = []
        score_per_feature = []
        metric_energy = []
        loss_energy = []

        try:
            # Prepare input for model
            model_input = torch.FloatTensor(data_window).unsqueeze(0)
            
            with torch.no_grad():
                output, series, prior, _ = self.model.model(model_input)
                
                # Calculate losses
                criterion = torch.nn.MSELoss(reduction='none')
                loss = torch.mean(criterion(model_input, output), dim=-1)
                loss_per_feature = criterion(model_input, output)
                
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
                loss = loss.detach().cpu().numpy()
                loss_per_feature = loss_per_feature.detach().cpu().numpy()  # Shape: (1, 100, 7)

                # Use the raw loss values from the latest timestep without any normalization
                latest_timestep_loss = loss_per_feature[0, -1, :]  # Shape: (7,) - latest timestep only
                
                # Anomaly score is the mean of raw losses at the latest timestep
                final_anomaly_score = float(np.mean(latest_timestep_loss))
                
                # Feature contributions are the raw loss values for each feature at the latest timestep
                feature_contributions = latest_timestep_loss  # Shape: (7,)
                
                return final_anomaly_score, feature_contributions
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return 0.0, np.zeros(len(self.feature_names))
                
    def _kl_loss(self, p, q):
        """KL divergence loss"""
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)
        
    def _calculate_feature_contributions(self, data_window: np.ndarray) -> np.ndarray:
        """Calculate feature contribution scores"""
        try:
            if len(data_window) >= 2:
                recent_mean = np.mean(data_window, axis=0)
                recent_std = np.std(data_window, axis=0) + 1e-8
                current_reading = data_window[-1]
                deviations = np.abs(current_reading - recent_mean) / recent_std
                return deviations
            return np.zeros(len(self.feature_names))
        except Exception:
            return np.zeros(len(self.feature_names))
            
    def is_anomaly(self, score: float) -> bool:
        """Check if score indicates an anomaly"""
        return score > self.threshold
