import pandas as pd
import numpy as np
import argparse
import logging


class ExtremeAnomalyInjector:
    """Inject extreme anomalies into weather data - realistic sensor failures"""
    
    def __init__(self, dataset_path: str, seed=None):
        """
        Initialize the anomaly injector
        
        Args:
            dataset_path: Path to original dataset.csv (will use last 2000 rows)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.logger = logging.getLogger(__name__)
        self.dataset_path = dataset_path
        
        # Load dataset
        self._load_data()
    
    def _load_data(self):
        """Load the last 2000 rows from dataset"""
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        
        # Take last 2000 rows
        if len(df) > 2000:
            df = df.tail(2000).reset_index(drop=True)
            self.logger.info(f"Using last 2000 rows from dataset")
        else:
            self.logger.info(f"Dataset has {len(df)} rows (less than 2000)")
            df = df.reset_index(drop=True)
        
        self.data = df.copy()
        
        # Identify feature columns (exclude date and is_anomaly if present)
        self.feature_cols = [col for col in self.data.columns if col not in ['date', 'is_anomaly']]
        
        # Add is_anomaly column if it doesn't exist
        if 'is_anomaly' not in self.data.columns:
            self.data['is_anomaly'] = 0
        
        # Calculate statistics
        self.feature_stats = {}
        for col in self.feature_cols:
            self.feature_stats[col] = {
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'mean': self.data[col].mean(),
                'std': self.data[col].std()
            }
        
        self.logger.info(f"Loaded {len(self.data)} rows with {len(self.feature_cols)} features")
    
    def _generate_extreme_anomaly(self, feature: str) -> float:
        """
        Generate anomalous values for specific weather features
        
        Args:
            feature: Feature name (tt, rh, pp, ws, wd, sr, rr)
            
        Returns:
            Anomalous value for that feature
        """
        stats = self.feature_stats[feature]
        
        # Feature-specific anomalies
        if feature == 'tt':  # Temperature
            # Temperature deviations: ±8-12°C from normal
            if np.random.rand() > 0.5:
                return stats['mean'] + np.random.uniform(8, 12)  # Heat spike
            else:
                return stats['mean'] - np.random.uniform(8, 12)  # Cold spike
        
        elif feature == 'rh':  # Relative Humidity
            # Extreme humidity: 0-10% (too dry) or 90-100% (too wet)
            if np.random.rand() > 0.5:
                return np.random.uniform(90, 100)  # Too wet
            else:
                return np.random.uniform(0, 10)    # Too dry
        
        elif feature == 'pp':  # Atmospheric Pressure
            # Pressure anomalies: high (1030+) or low (930-)
            if np.random.rand() > 0.5:
                return np.random.uniform(1030, 1040)  # High
            else:
                return np.random.uniform(920, 930)    # Low
        
        elif feature == 'ws':  # Wind Speed
            # Wind speed anomaly: 6-12 m/s (elevated but not extreme)
            return np.random.uniform(6, 12)
        
        elif feature == 'wd':  # Wind Direction
            # Invalid values to represent sensor failure
            invalid_values = [-999, 0, 999]
            return float(np.random.choice(invalid_values))
        
        elif feature == 'sr':  # Solar Radiation
            # Solar radiation: 0 (night/failure) or 500-900 (midday saturation)
            if np.random.rand() > 0.5:
                return np.random.uniform(500, 900)  # Elevated radiation
            else:
                return 0.0  # No radiation / sensor failure
        
        elif feature == 'rr':  # Rainfall
            # Rainfall anomaly: 2-8 mm (notable rain)
            return np.random.uniform(2, 8)
        
        else:
            # Default: push to upper range
            return stats['mean'] + stats['std']
    
    def inject_anomalies(self, n_anomalies: int, min_duration: int = 1, 
                        max_duration: int = 10) -> pd.DataFrame:
        """
        Inject extreme anomalies into the dataset
        
        Args:
            n_anomalies: Number of anomalous events to inject
            min_duration: Minimum duration of each anomaly
            max_duration: Maximum duration of each anomaly
            
        Returns:
            DataFrame with injected anomalies and is_anomaly column
        """
        df = self.data.copy()
        
        self.logger.info(f"Injecting {n_anomalies} extreme anomalies with duration {min_duration}-{max_duration}")
        
        # Calculate available positions
        available_indices = list(range(len(df) - max_duration))
        
        if len(available_indices) < n_anomalies:
            n_anomalies = len(available_indices)
            self.logger.warning(f"Reducing anomalies to {n_anomalies} due to space constraints")
        
        # Randomly select non-overlapping anomaly positions
        anomaly_positions = sorted(np.random.choice(available_indices, n_anomalies, replace=False))
        
        anomaly_indices = set()
        
        # Inject anomalies
        for start_pos in anomaly_positions:
            duration = np.random.randint(min_duration, max_duration + 1)
            duration = min(duration, len(df) - start_pos)
            
            # Randomly select 1-2 features to corrupt
            n_features = np.random.randint(1, 3)
            features_to_corrupt = np.random.choice(self.feature_cols, n_features, replace=False)
            
            for i in range(start_pos, start_pos + duration):
                # Generate extreme anomalies for selected features
                for feature in features_to_corrupt:
                    anomalous_value = self._generate_extreme_anomaly(feature)
                    df.at[i, feature] = anomalous_value
                
                df.at[i, 'is_anomaly'] = 1
                anomaly_indices.add(i)
        
        # Log statistics
        n_anomaly_samples = len(anomaly_indices)
        anomaly_ratio = n_anomaly_samples / len(df) * 100
        
        self.logger.info(f"Injected {n_anomaly_samples} anomalous samples ({anomaly_ratio:.2f}%)")
        
        return df


def main():
    """Main function with CLI"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    parser = argparse.ArgumentParser(description='Inject extreme anomalies into weather dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to original dataset.csv')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--n-anomalies', type=int, default=50, help='Number of anomalous events to inject')
    parser.add_argument('--min-duration', type=int, default=1, help='Minimum anomaly duration (samples)')
    parser.add_argument('--max-duration', type=int, default=10, help='Maximum anomaly duration (samples)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create injector
    injector = ExtremeAnomalyInjector(args.dataset, seed=args.seed)
    
    # Inject anomalies
    df = injector.inject_anomalies(
        n_anomalies=args.n_anomalies,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    # Save to file
    df.to_csv(args.output, index=False)
    
    print(f"\n✓ Data with extreme anomalies saved to {args.output}")
    print(f"  Total samples: {len(df)}")
    print(f"  Anomalous samples: {df['is_anomaly'].sum()}")
    print(f"  Anomaly percentage: {df['is_anomaly'].sum()/len(df)*100:.2f}%")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string())
    print(f"\nFirst 10 anomalous rows:")
    anomalous = df[df['is_anomaly'] == 1].head(10)
    if len(anomalous) > 0:
        print(anomalous.to_string())
    else:
        print("No anomalous rows")


if __name__ == '__main__':
    main()