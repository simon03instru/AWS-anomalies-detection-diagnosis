# Updated data_processor.py
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from data_storage import WeatherDataStorage
from scaler_manager import WeatherDataScaler

class PersistentDataProcessor:
    """Data processor that reads from persistent storage instead of memory buffer"""
    
    def __init__(self, feature_names: List[str], win_size: int = 100, 
                 storage: WeatherDataStorage = None, scaler: Optional[object] = None):
        self.feature_names = feature_names
        self.win_size = win_size
        self.scaler = scaler
        self.storage = storage or WeatherDataStorage()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with historical data from dataset
        self._initialize_with_historical_data()
        
    def _initialize_with_historical_data(self):
        """Initialize storage with historical data from dataset if empty"""
        try:
            current_count = self.storage.get_data_count()
            
            if current_count < self.win_size:
                self.logger.info(f"Storage has only {current_count} records, loading historical data...")
                self._load_historical_data()
            else:
                self.logger.info(f"Storage already has {current_count} records, skipping historical load")
                
        except Exception as e:
            self.logger.error(f"Error initializing with historical data: {e}")
    
    def _load_historical_data(self):
        """Load last 100 records from dataset/dataset_fixx_dki_staklim.csv to populate storage"""
        try:
            dataset_path = "dataset/dataset_fix/diy_pakem.csv"
            
            if not os.path.exists(dataset_path):
                self.logger.warning(f"Historical dataset not found: {dataset_path}")
                # Try alternative path
                alt_path = "dataset/dataset_fix/diy_pakem.csv"
                if os.path.exists(alt_path):
                    dataset_path = alt_path
                    self.logger.info(f"Using alternative dataset: {dataset_path}")
                else:
                    self.logger.error(f"No dataset found at {dataset_path} or {alt_path}")
                    return
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset: {len(df)} total rows")
            
            # Take last 100 rows to populate storage
            last_100 = df.tail(100).copy()
            self.logger.info(f"Taking last 100 rows from dataset")
            
            # Clean the data - handle missing values (-999.0)
            last_100 = last_100.replace(-999.0, np.nan)
            last_100 = last_100.fillna(method='ffill').fillna(method='bfill')
            
            # Show what we're working with
            print(f"Dataset columns: {list(df.columns)}")
            print(f"Required features: {self.feature_names}")
            
            # Generate timestamps (going backwards from 10 minutes ago to avoid conflicts with real-time data)
            base_time = datetime.now() - timedelta(minutes=10 + len(last_100) * 5)
            
            count = 0
            for i, (_, row) in enumerate(last_100.iterrows()):
                timestamp = base_time + timedelta(minutes=i * 5)  # 5-minute intervals
                
                # Extract weather data
                weather_data = {}
                for feature in self.feature_names:
                    if feature in row and not pd.isna(row[feature]):
                        weather_data[feature] = float(row[feature])
                    else:
                        # Use fallback values for missing features
                        fallback = {
                            'tt': 27.0, 'rh': 75.0, 'pp': 1013.0, 'ws': 5.0,
                            'wd': 180.0, 'sr': 400.0, 'rr': 0.0
                        }
                        weather_data[feature] = fallback.get(feature, 0.0)
                        if feature in row:
                            self.logger.warning(f"Missing {feature} at row {i}, using fallback: {weather_data[feature]}")
                
                # Save to storage
                if self.storage.save_weather_data(weather_data, timestamp):
                    count += 1
            
            self.logger.info(f"Successfully loaded {count} historical records into storage")
            print(f"Database initialized with last 100 records from {dataset_path}")
            
            # Show sample of loaded data
            recent_data = self.storage.get_recent_data(limit=3)
            if len(recent_data) > 0:
                print("Sample loaded data:")
                for _, row in recent_data.iterrows():
                    sample_str = f"  {row['timestamp']}: "
                    sample_str += " | ".join([f"{feat}:{row[feat]:.1f}" for feat in self.feature_names if feat in row])
                    print(sample_str)
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            print(f"Failed to load historical data: {e}")
            print("System will start with empty storage and build data as MQTT messages arrive")
    
    def add_reading(self, reading: Dict, timestamp: datetime) -> bool:
        """
        Add new reading to persistent storage
        
        Args:
            reading: Weather data dictionary
            timestamp: Timestamp of the reading
            
        Returns:
            True if we have enough data for anomaly detection
        """
        try:
            # Save to persistent storage
            success = self.storage.save_weather_data(reading, timestamp)
            
            if not success:
                self.logger.error("Failed to save weather data to storage")
                return False
            
            # Check if we have enough data
            total_count = self.storage.get_data_count()
            has_enough = total_count >= self.win_size
            
            self.logger.debug(f"Saved reading, total records: {total_count}, enough for detection: {has_enough}")
            
            return has_enough
            
        except Exception as e:
            self.logger.error(f"Error adding reading: {e}")
            return False
    
    def get_current_window(self) -> Optional[np.ndarray]:
        """Get current data window from persistent storage"""
        try:
            # Get recent data from storage
            df = self.storage.get_recent_data(limit=self.win_size)
            
            if len(df) < self.win_size:
                self.logger.warning(f"Not enough data in storage: {len(df)}/{self.win_size}")
                return None
            
            # Extract feature values in correct order
            feature_data = []
            for feature in self.feature_names:
                if feature in df.columns:
                    values = df[feature].values
                    # Handle NaN values
                    values = np.nan_to_num(values, nan=0.0)
                    feature_data.append(values)
                else:
                    self.logger.warning(f"Feature {feature} not found in storage data")
                    feature_data.append(np.zeros(len(df)))
            
            # Convert to numpy array (features, time) -> (time, features)
            data_window = np.column_stack(feature_data)
            
            # Scale if scaler available
            if self.scaler:
                data_window = self.scaler.scale_data(data_window)
            
            self.logger.debug(f"Created data window: {data_window.shape}")
            
            return data_window
            
        except Exception as e:
            self.logger.error(f"Error getting current window: {e}")
            return None
    
    def get_current_timestamp(self) -> Optional[datetime]:
        """Get the most recent timestamp from storage"""
        try:
            df = self.storage.get_recent_data(limit=1)
            if len(df) > 0:
                return df['timestamp'].iloc[-1]
            return None
        except:
            return None
    
    def get_storage_status(self) -> Dict:
        """Get storage status information"""
        return {
            'total_records': self.storage.get_data_count(),
            'database_path': self.storage.db_path,
            'csv_path': self.storage.csv_path,
            'has_enough_data': self.storage.get_data_count() >= self.win_size
        }
