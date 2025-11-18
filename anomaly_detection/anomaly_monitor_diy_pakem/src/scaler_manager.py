# scaler_manager.py
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import RobustScaler
from typing import Optional, List
import logging

class WeatherDataScaler:
    """Handles scaling of weather data using RobustScaler fitted to training data"""
    
    def __init__(self, dataset_path: str, feature_names: List[str], scaler_save_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.feature_names = feature_names
        self.scaler_save_path = scaler_save_path or os.path.join(os.path.dirname(dataset_path), 'weather_scaler.pkl')
        self.scaler = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaler
        self._load_or_fit_scaler()
        
    def _load_or_fit_scaler(self):
        """Load existing scaler or fit new one to training data"""
        
        # Try to load existing scaler first
        if os.path.exists(self.scaler_save_path):
            try:
                with open(self.scaler_save_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Loaded existing scaler from {self.scaler_save_path}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load existing scaler: {e}")
        
        # Fit new scaler to training data
        self._fit_scaler_to_dataset()
        
    def _fit_scaler_to_dataset(self):
        """Fit RobustScaler to the training dataset"""
        try:
            self.logger.info(f"Fitting RobustScaler to dataset: {self.dataset_path}")
            
            # Load the dataset
            df = pd.read_csv(self.dataset_path)
            
            # Map feature names to dataset columns
            feature_mapping = self._get_feature_mapping(df.columns.tolist())
            
            # Extract feature data in the correct order
            feature_data = []
            for feature in self.feature_names:
                if feature in feature_mapping:
                    column_name = feature_mapping[feature]
                    if column_name in df.columns:
                        # Handle missing values (-999.0)
                        column_data = df[column_name].replace(-999.0, np.nan)
                        # Forward fill missing values
                        column_data = column_data.fillna(method='ffill').fillna(method='bfill')
                        feature_data.append(column_data.values)
                    else:
                        self.logger.warning(f"Column {column_name} not found in dataset")
                        # Use dummy data
                        feature_data.append(np.full(len(df), 0.0))
                else:
                    self.logger.warning(f"No mapping found for feature {feature}")
                    # Use dummy data
                    feature_data.append(np.full(len(df), 0.0))
            
            # Convert to numpy array
            X = np.column_stack(feature_data)
            
            # Remove any remaining NaN or infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Fit the scaler
            self.scaler = RobustScaler()
            self.scaler.fit(X)
            
            # Save the scaler
            os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
            with open(self.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            self.logger.info(f"RobustScaler fitted and saved to {self.scaler_save_path}")
            self.logger.info(f"Scaler statistics:")
            self.logger.info(f"  Center (median): {self.scaler.center_}")
            self.logger.info(f"  Scale (IQR): {self.scaler.scale_}")
            
        except Exception as e:
            self.logger.error(f"Error fitting scaler to dataset: {e}")
            # Fallback to identity scaler
            self._create_identity_scaler()
            
    def _get_feature_mapping(self, dataset_columns: List[str]) -> dict:
        """Map feature names to dataset column names"""
        
        # Common mappings between feature names and dataset columns
        mapping = {}
        
        for feature in self.feature_names:
            # Direct match first
            if feature in dataset_columns:
                mapping[feature] = feature
            # Try common aliases
            elif feature == 'tt' or feature == 'temperature':
                for col in ['tt', 'temperature', 'temp', 'Temperature']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
            elif feature == 'rh' or feature == 'humidity':
                for col in ['rh', 'humidity', 'Humidity', 'RH']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
            elif feature == 'pp' or feature == 'pressure':
                for col in ['pp', 'pressure', 'Pressure', 'press']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
            elif feature == 'ws' or feature == 'wind_speed':
                for col in ['ws', 'wind_speed', 'windspeed', 'WindSpeed']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
            elif feature == 'wd' or feature == 'wind_direction':
                for col in ['wd', 'wind_direction', 'winddir', 'WindDirection']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
            elif feature == 'sr' or feature == 'solar_radiation':
                for col in ['sr', 'solar_radiation', 'solar', 'radiation']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
            elif feature == 'rr' or feature == 'rainfall':
                for col in ['rr', 'rainfall', 'rain', 'precipitation']:
                    if col in dataset_columns:
                        mapping[feature] = col
                        break
        
        self.logger.info(f"Feature mapping: {mapping}")
        return mapping
        
    def _create_identity_scaler(self):
        """Create an identity scaler (no scaling) as fallback"""
        self.logger.warning("Creating identity scaler as fallback")
        self.scaler = RobustScaler()
        # Fit with dummy data that results in no scaling
        dummy_data = np.random.normal(0, 1, (100, len(self.feature_names)))
        self.scaler.fit(dummy_data)
        # Override to make it identity
        self.scaler.center_ = np.zeros(len(self.feature_names))
        self.scaler.scale_ = np.ones(len(self.feature_names))
        
    def scale_data(self, data: np.ndarray) -> np.ndarray:
        """Scale incoming data using the fitted scaler"""
        try:
            if self.scaler is None:
                self.logger.error("Scaler not initialized")
                return data
                
            # Handle NaN and infinite values
            data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale the data
            scaled_data = self.scaler.transform(data_clean)
            
            return scaled_data
            
        except Exception as e:
            self.logger.error(f"Error scaling data: {e}")
            return data
            
    def scale_single_sample(self, sample: List[float]) -> List[float]:
        """Scale a single data sample"""
        try:
            # Convert to numpy array and reshape for single sample
            sample_array = np.array(sample).reshape(1, -1)
            scaled_array = self.scale_data(sample_array)
            return scaled_array[0].tolist()
        except Exception as e:
            self.logger.error(f"Error scaling single sample: {e}")
            return sample

