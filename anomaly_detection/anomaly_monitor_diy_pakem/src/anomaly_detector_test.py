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

class AnomalyDetectionEngine:
    """Core anomaly detection engine using the trained AnomalyTransformer model (without data loading)"""
    
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
        """Load only the AnomalyTransformer model weights without initializing Trainer"""
        try:
            from AnomalyTransformer import AnomalyTransformer
            
            # Get input/output channels from feature names
            input_channels = len(self.feature_names)
            output_channels = len(self.feature_names)
            
            # Initialize model with standard parameters
            self.model = AnomalyTransformer(
                win_size=self.win_size,
                enc_in=input_channels,
                c_out=output_channels,
                e_layers=3
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's a raw state dict
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.logger.info(f"AnomalyTransformer model loaded successfully from {self.checkpoint_path}")
            self.logger.info(f"Model config: win_size={self.win_size}, enc_in={input_channels}, c_out={output_channels}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import AnomalyTransformer: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    def _set_threshold(self):
        """Set the anomaly threshold"""
        if self.use_fixed_threshold:
            self.threshold = self.fixed_threshold
            self.logger.info(f"Using FIXED threshold: {self.threshold}")
        else:
            self.threshold = self.fixed_threshold
            self.logger.warning(f"Dynamic threshold calculation skipped. Using fixed threshold: {self.threshold}")
    
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
        try:
            # Prepare input for model: add batch dimension
            # Input shape: (win_size, n_features) -> (1, win_size, n_features)
            model_input = torch.FloatTensor(data_window).unsqueeze(0)
            
            with torch.no_grad():
                # Get model outputs
                output, series, prior, _ = self.model(model_input)
                
                # Calculate reconstruction loss per feature
                criterion = nn.MSELoss(reduction='none')
                loss_per_feature = criterion(model_input, output)  # Shape: (1, win_size, n_features)
                
                # Extract loss for the latest timestep
                # Shape: (1, win_size, n_features) -> (n_features,)
                latest_timestep_loss = loss_per_feature[0, -1, :]
                
                # Anomaly score is the mean of losses at the latest timestep
                final_anomaly_score = float(torch.mean(latest_timestep_loss).item())
                
                # Feature contributions are individual feature losses at the latest timestep
                feature_contributions = latest_timestep_loss.detach().cpu().numpy()
                
                return final_anomaly_score, feature_contributions
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            import traceback
            traceback.print_exc()
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