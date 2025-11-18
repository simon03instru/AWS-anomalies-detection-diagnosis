import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset

class WeatherSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", target_column='tt', label_column=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.target_column = target_column
        self.label_column = label_column
        self.scaler = RobustScaler()

        # Column mapping for weather data
        self.column_mapping = {
            'tt': 'Air Temperature (°C)',
            'rh': 'Relative Humidity (%)',
            'rr': 'Rainfall (mm)',
            'ws': 'Wind Speed (m/s)',
            'wd': 'Wind Direction (degrees)',
            'sr': 'Solar Radiation (W/m²)',
            'pp': 'Air Pressure (hPa)'
        }

        # Load single data file
        try:
            if data_path.endswith('.csv'):
                full_data = pd.read_csv(data_path)
            else:
                import os
                possible_files = ['data.csv', 'weather.csv', 'dataset.csv', 'full_data.csv']
                file_found = False
                for filename in possible_files:
                    file_path = os.path.join(data_path, filename)
                    if os.path.exists(file_path):
                        full_data = pd.read_csv(file_path)
                        file_found = True
                        print(f"Found and loaded: {file_path}")
                        break
                
                if not file_found:
                    raise FileNotFoundError(f"No data file found in {data_path}. Please ensure your file is named one of: {possible_files} or provide the full file path.")
        
        except Exception as e:
            raise FileNotFoundError(f"Could not load data from {data_path}. Error: {str(e)}")

        print(f"Loaded data with shape: {full_data.shape}")

        # Select features - exclude date columns automatically
        date_columns = []
        for col in full_data.columns:
            col_lower = col.lower()
            if any(date_keyword in col_lower for date_keyword in ['date', 'time', 'timestamp', 'datetime']):
                date_columns.append(col)
        
        if date_columns:
            print(f"Automatically excluding date columns: {date_columns}")
        
        # Get all non-date columns
        feature_columns = [col for col in full_data.columns if col not in date_columns]
        
        if target_column == 'all':
            if not feature_columns:
                raise ValueError("No feature columns found after excluding date columns")
            data = full_data[feature_columns].values
            self.selected_columns = feature_columns
            print(f"Using all {len(feature_columns)} feature columns: {feature_columns}")
        elif target_column in full_data.columns:
            if target_column in date_columns:
                raise ValueError(f"Cannot use date column '{target_column}' as target")
            data = full_data[[target_column]].values
            self.selected_columns = [target_column]
            print(f"Using single target column: {target_column}")
        else:
            raise ValueError(f"Column '{target_column}' not found in data. Available columns: {list(full_data.columns)}")

        # Split data into train (75%), val (15%), test (10%) BEFORE cleaning
        total_len = len(data)
        train_end = int(0.75 * total_len)
        val_end = int(0.90 * total_len)
        
        train_data = data[:train_end].copy()
        val_data = data[train_end:val_end].copy()
        test_data = data[val_end:].copy()

        print(f"Original data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Clean training data (remove NaN and outliers)
        train_data_clean = self._remove_nan_and_outliers(train_data, "Training")
        
        # Clean validation data (remove NaN and outliers)
        val_data_clean = self._remove_nan_and_outliers(val_data, "Validation")
        
        # Test data: only handle NaN values, keep outliers
        test_data_clean = np.nan_to_num(test_data)
        print(f"Test set: Only NaN values replaced with 0, outliers preserved")

        self.train = train_data_clean
        self.val = val_data_clean
        self.test = test_data_clean

        # === Scaling step ===
        print("\nApplying RobustScaler...")
        self.train = self.scaler.fit_transform(self.train)
        self.val = self.scaler.transform(self.val)
        self.test = self.scaler.transform(self.test)

        print(f"Final data shapes - Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")

        # Handle labels with same cleaning approach
        if label_column and label_column in full_data.columns:
            if label_column in date_columns:
                print(f"Warning: Label column '{label_column}' appears to be a date column. Using as numeric label anyway.")
            label_data = full_data[[label_column]].values.astype(np.float32)
            
            # Split labels
            train_labels = label_data[:train_end]
            val_labels = label_data[train_end:val_end]
            test_labels = label_data[val_end:]
            
            # Clean labels (same rows as data)
            if len(train_data_clean) < len(train_labels):
                # If we removed outliers from train data, we need to remove corresponding labels
                train_mask = self._get_outlier_mask(train_data)
                self.train_labels = train_labels[train_mask]
            else:
                self.train_labels = train_labels
            
            if len(val_data_clean) < len(val_labels):
                # If we removed outliers from val data, we need to remove corresponding labels
                val_mask = self._get_outlier_mask(val_data)
                self.val_labels = val_labels[val_mask]
            else:
                self.val_labels = val_labels
                
            # Test labels: only handle NaN
            self.test_labels = np.nan_to_num(test_labels)
        else:
            # Create dummy labels
            self.train_labels = np.zeros((self.train.shape[0], 1), dtype=np.float32)
            self.val_labels = np.zeros((self.val.shape[0], 1), dtype=np.float32)
            self.test_labels = np.zeros((self.test.shape[0], 1), dtype=np.float32)

        print(f"Target column: {target_column}")
        print(f"Label shapes - Train: {self.train_labels.shape}, Val: {self.val_labels.shape}, Test: {self.test_labels.shape}")

    def _remove_nan_and_outliers(self, data, dataset_name):
        """Remove NaN values and outliers using 1.5 IQR method"""
        print(f"\nCleaning {dataset_name} data:")
        original_shape = data.shape
        
        # Step 1: Remove rows with NaN values
        nan_mask = ~np.isnan(data).any(axis=1)
        data_no_nan = data[nan_mask]
        nan_removed = original_shape[0] - len(data_no_nan)
        print(f"  Removed {nan_removed} rows with NaN values")
        
        # Step 2: Remove outliers using 1.5 IQR method
        outlier_mask = np.ones(len(data_no_nan), dtype=bool)
        
        for col_idx in range(data_no_nan.shape[1]):
            col_data = data_no_nan[:, col_idx]
            
            # Calculate Q1, Q3, and IQR
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Update mask (keep rows that are NOT outliers in this column)
            col_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
            outlier_mask = outlier_mask & col_mask
            
            col_outliers = np.sum(~col_mask)
            if col_outliers > 0:
                col_name = self.selected_columns[col_idx] if hasattr(self, 'selected_columns') and col_idx < len(self.selected_columns) else f"Column_{col_idx}"
                print(f"    {col_name}: {col_outliers} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")
        
        data_clean = data_no_nan[outlier_mask]
        outliers_removed = len(data_no_nan) - len(data_clean)
        total_removed = original_shape[0] - len(data_clean)
        
        print(f"  Removed {outliers_removed} rows with outliers")
        print(f"  Total removed: {total_removed} rows ({total_removed/original_shape[0]:.1%})")
        print(f"  Final {dataset_name.lower()} shape: {data_clean.shape}")
        
        return data_clean
    
    def _get_outlier_mask(self, data):
        """Get mask for non-outlier rows (used for label alignment)"""
        # Remove NaN rows first
        nan_mask = ~np.isnan(data).any(axis=1)
        data_no_nan = data[nan_mask]
        
        # Get outlier mask
        outlier_mask = np.ones(len(data_no_nan), dtype=bool)
        
        for col_idx in range(data_no_nan.shape[1]):
            col_data = data_no_nan[:, col_idx]
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
            outlier_mask = outlier_mask & col_mask
        
        # Combine masks
        final_mask = np.zeros(len(data), dtype=bool)
        final_mask[nan_mask] = outlier_mask
        
        return final_mask

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __getitem__(self, index):
        start_idx = index * self.step
        if self.mode == "train":
            return (
                np.float32(self.train[start_idx:start_idx + self.win_size]),
                np.float32(self.train_labels[start_idx:start_idx + self.win_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[start_idx:start_idx + self.win_size]),
                np.float32(self.val_labels[start_idx:start_idx + self.win_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[start_idx:start_idx + self.win_size]),
                np.float32(self.test_labels[start_idx:start_idx + self.win_size])
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='all', label_column=None):
    """
    Create data loader for weather anomaly detection from a single file
    Note: Training and validation sets will have NaN values and outliers (1.5 IQR) removed
          Test set will only have NaN values replaced with 0, outliers preserved

    Args:
        data_path: Path to single CSV file or directory containing the data file
        batch_size: Batch size for training
        win_size: Window size for time series segments
        step: Step size for sliding window
        mode: 'train', 'val', or 'test'
        dataset: Target column name or 'all'
        label_column: Label column name (optional)
    """
    if mode not in ['train', 'val', 'test']:
        raise ValueError(f"Unsupported mode: {mode}. Only 'train', 'val', and 'test' are supported.")

    shuffle = (mode == 'train')

    dataset_obj = WeatherSegLoader(
        data_path=data_path,
        win_size=win_size,
        step=step,
        mode=mode,
        target_column=dataset,
        label_column=label_column
    )

    return DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


# Example usage:
"""
# Load training data - will remove NaN and outliers
train_loader = get_loader_segment(
    data_path='./data/weather_data.csv',
    batch_size=32, 
    win_size=100, 
    step=1, 
    mode='train', 
    dataset='all'
)

# Load validation data - will remove NaN and outliers
val_loader = get_loader_segment(
    data_path='./data/weather_data.csv',
    batch_size=32, 
    win_size=100, 
    step=1, 
    mode='val', 
    dataset='all'
)

# Load test data - only replaces NaN with 0, keeps outliers
test_loader = get_loader_segment(
    data_path='./data/weather_data.csv',
    batch_size=32, 
    win_size=100, 
    step=1, 
    mode='test', 
    dataset='all'
)
"""