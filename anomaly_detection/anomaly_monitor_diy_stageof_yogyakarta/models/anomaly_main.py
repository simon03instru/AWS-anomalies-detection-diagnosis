import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils import *
from AnomalyTransformer import AnomalyTransformer
from data_loader import get_loader_segment
from mlflowmanager import MLflowManager


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = os.path.join(path, str(self.dataset) + '_checkpoint.pth')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        # Log to MLflow
        if hasattr(self, "mlflow_manager") and self.mlflow_manager:
            self.mlflow_manager.log_artifact(save_path, artifact_path="checkpoints")

class Trainer():
    DEFAULTS = {
        'data_path': 'dataset/dataset_fix/dki_staklim.csv',
        'batch_size': 32,
        'win_size': 100,
        'step' : 100,
        'input_c': 7,
        'output_c': 7,
        'dataset': 'all',
        'lr': 0.001,
        'num_epochs': 10,
        'k': 0.1,
        'model_save_path': './checkpoints/dki_staklim/',
        'anomaly_ratio': 0.1,  # Percentage of anomalies to detect
    }

    def __init__(self, config, mlflow_manager=None):
        # Merge defaults and user config
        full_config = {**self.DEFAULTS, **config}
        self.__dict__.update(full_config)
        self.mlflow_manager = MLflowManager()

        # Only override device if it's not provided in config
        if 'device' not in config:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load data
        self.train_loader = get_loader_segment(self.data_path, self.batch_size, self.win_size, step = self.step, mode = 'train', dataset = self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, self.batch_size, self.win_size, step = self.step, mode = 'val', dataset = self.dataset)
        self.test_loader = get_loader_segment(self.data_path, self.batch_size, self.win_size, step = self.step, mode = 'test', dataset = self.dataset)
        #self.thre_loader = get_loader_segment(self.data_path, self.batch_size, self.win_size, 'thre', self.dataset)

        # Build model and loss
        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            if self.mlflow_manager:
                self.mlflow_manager.log_metric("train_loss", train_loss, step=epoch)
                self.mlflow_manager.log_metric("val_loss_1", vali_loss1, step=epoch)
                self.mlflow_manager.log_metric("val_loss_2", vali_loss2, step=epoch)

    def test(self):
        """
        Modified test method that keeps scalar threshold but provides feature attribution
        for detected anomalies
        """
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
                map_location=torch.device("cpu")   # <--- force CPU
            )
        )
        self.model.eval()
        temperature = 50

        print("======================TEST MODE ======================")

        criterion = nn.MSELoss(reduce=False)
        
        # (1) Statistics on the train set - sample level (original approach)
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) Find the threshold - sample level (original approach)
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        #thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)

        print("Threshold :", thresh)
        return combined_energy, thresh

    def detect(self, test_loader=None, threshold=None):
        """
        Detect anomalies using scalar threshold and provide feature attribution for detected anomalies
        """
        if test_loader is None:
            # Create test loader if not provided
            test_loader = self.test_loader
        if threshold is None:
            threshold = self.test()
        
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
                map_location=torch.device("cpu")   # <--- force CPU
            )
        )
        self.model.eval()
        temperature = 50

        print("======================DETECTION MODE ======================")
        
        criterion = nn.MSELoss(reduction='none')
        attens_energy = []
        score_per_feature = []
        metric_energy = []
        loss_energy = []

        for i, (input_data, labels) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)
            loss_per_feature = criterion(input, output)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            
            # Normalize metric and loss before calculating energy
            metric = (metric - torch.min(metric)) / (torch.max(metric) - torch.min(metric) + 1e-8)
            loss = (loss - torch.min(loss)) / (torch.max(loss) - torch.min(loss) + 1e-8)
            # Calculate energy
            # Note: Ensure metric and loss are not zero to avoid NaN in energy calculation
            
            metric = torch.clamp(metric, min=1e-8, max=1.0)
            loss = torch.clamp(loss, min=1e-8, max=1.0)
            # Calculate energy as product of metric and loss
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            loss_per_feature = loss_per_feature.detach().cpu().numpy()
            metric = metric.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            attens_energy.append(cri)
            metric_energy.append(metric)
            score_per_feature.append(loss_per_feature)
            loss_energy.append(loss)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        score_per_feature = np.concatenate(score_per_feature, axis=0).reshape(-1, self.win_size, self.input_c)
        loss_energy = np.concatenate(loss_energy, axis=0).reshape(-1)
        metric_energy = np.concatenate(metric_energy, axis=0).reshape(-1)

        test_energy = np.array(attens_energy)
        
        #return score_per_feature, test_energy, metric_energy, loss_energy
        return score_per_feature, loss_energy
    

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import os

class AnomalyDetector:
    def __init__(self, model_instance):
        """
        Initialize the anomaly detector with your model instance
        
        Args:
            model_instance: Your existing model class instance that has the detect method
        """
        self.model = model_instance
    
    def extract_features_and_dates(self, data_path: str, test_split: float = 0.1) -> Tuple[List[str], pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Extract feature names and dates from the CSV file
        
        Args:
            data_path: Path to the CSV file
            test_split: Fraction of data used for testing (default: 0.1 for last 10%)
            
        Returns:
            Tuple of (feature_names, all_dates, test_dates)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        print(f"Full dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Calculate test data boundaries (last 10%)
        total_samples = len(df)
        test_start_idx = int(total_samples * (1 - test_split))
        test_samples = total_samples - test_start_idx
        
        print(f"Total samples: {total_samples}")
        print(f"Test start index: {test_start_idx}")
        print(f"Test samples: {test_samples}")
        
        # Try to identify date/time column
        date_column = None
        possible_date_columns = ['date', 'datetime', 'timestamp', 'time', 'Date', 'DateTime', 'Timestamp', 'Time']
        
        for col in possible_date_columns:
            if col in df.columns:
                date_column = col
                break
        
        # If no obvious date column, check for columns that might contain dates
        if date_column is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head())
                        date_column = col
                        print(f"Detected date column: {col}")
                        break
                    except:
                        continue
        
        # Extract dates
        if date_column is not None:
            try:
                all_dates = pd.to_datetime(df[date_column])
                test_dates = all_dates[test_start_idx:]
                print(f"Successfully parsed dates from column: {date_column}")
                print(f"Full date range: {all_dates.min()} to {all_dates.max()}")
                print(f"Test date range: {test_dates.min()} to {test_dates.max()}")
            except Exception as e:
                print(f"Error parsing dates from {date_column}: {e}")
                print("Creating artificial date range...")
                all_dates = pd.date_range('2024-01-01', periods=total_samples, freq='1H')
                test_dates = all_dates[test_start_idx:]
        else:
            print("No date column found. Creating artificial date range...")
            all_dates = pd.date_range('2024-01-01', periods=total_samples, freq='1H')
            test_dates = all_dates[test_start_idx:]
        
        # Extract feature names (exclude date column and any label columns)
        feature_columns = [col for col in df.columns if col != date_column]
        
        # Remove common label columns
        label_keywords = ['label', 'target', 'class', 'anomaly', 'Label', 'Target', 'Class', 'Anomaly']
        feature_columns = [col for col in feature_columns if col not in label_keywords]
        
        print(f"Identified {len(feature_columns)} features: {feature_columns}")
        
        return feature_columns, all_dates, test_dates
        
    def detect_anomalies_with_attribution(self, 
                                        test_loader=None, 
                                        threshold=None,
                                        percentile_threshold=95,
                                        data_path=None,
                                        test_split=0.1,
                                        win_size=100,
                                        step=100,
                                        window_size=5) -> Dict:
        """
        Detect anomalies and provide feature attribution
        
        Args:
            test_loader: Test data loader
            threshold: Fixed threshold value (if None, uses percentile)
            percentile_threshold: Percentile for automatic threshold (default: 95)
            data_path: Path to CSV file to extract feature names and dates
            test_split: Fraction of data used for testing (default: 0.1 for last 10%)
            win_size: Window size used in data loader (default: 100)
            step: Step size used in data loader (default: 100)
            window_size: Time window around anomaly to extract (±window_size)
            
        Returns:
            Dictionary containing anomaly results and feature attributions
        """
        # Extract feature names and dates from CSV if path provided
        if data_path is not None:
            feature_names, all_dates, test_dates = self.extract_features_and_dates(data_path, test_split)
        else:
            feature_names = None
            all_dates = None
            test_dates = None
        
        # Get anomaly scores from your existing method
        score_per_feature, loss_energy = self.model.detect(test_loader, threshold)
        
        print(f"Model output - score_per_feature shape: {score_per_feature.shape}")
        print(f"Model output - loss_energy shape: {loss_energy.shape}")
        
        # Handle the shape mismatch:
        # score_per_feature: (batch_size, win_size, n_features) e.g., (52, 100, 8)
        # loss_energy: (batch_size * win_size,) e.g., (5200,) - flattened
        
        if len(score_per_feature.shape) == 3 and len(loss_energy.shape) == 1:
            batch_size, win_size_actual, n_features = score_per_feature.shape
            expected_flattened_size = batch_size * win_size_actual
            
            print(f"Detected shape pattern:")
            print(f"  Batch size: {batch_size}")
            print(f"  Window size: {win_size_actual}")
            print(f"  Number of features: {n_features}")
            print(f"  Expected flattened size: {expected_flattened_size}")
            print(f"  Actual loss_energy size: {len(loss_energy)}")
            
            if len(loss_energy) == expected_flattened_size:
                # Reshape score_per_feature to (5200, 8) to match loss_energy
                score_per_feature_flattened = score_per_feature.reshape(-1, n_features)
                
                print(f"Reshaped score_per_feature from {score_per_feature.shape} to {score_per_feature_flattened.shape}")
                print(f"Now both have matching first dimension: {len(score_per_feature_flattened)} timesteps")
                
                # Update variables for the rest of the function
                score_per_feature = score_per_feature_flattened
                
                print(f"Final score_per_feature shape: {score_per_feature.shape}")  # Should be (5200, 8)
                print(f"Final loss_energy shape: {loss_energy.shape}")  # Should be (5200,)
                
            else:
                print(f"WARNING: Unexpected size mismatch.")
                print(f"Expected: {expected_flattened_size}, Got: {len(loss_energy)}")
                # Fallback: truncate to minimum length
                min_length = min(expected_flattened_size, len(loss_energy))
                score_per_feature_flattened = score_per_feature.reshape(-1, n_features)[:min_length]
                loss_energy = loss_energy[:min_length]
                score_per_feature = score_per_feature_flattened
        
        # Verify that both outputs now have the same length
        if len(score_per_feature) != len(loss_energy):
            print(f"WARNING: Still have mismatch after reshaping.")
            print(f"score_per_feature length: {len(score_per_feature)}")
            print(f"loss_energy length: {len(loss_energy)}")
            min_length = min(len(score_per_feature), len(loss_energy))
            score_per_feature = score_per_feature[:min_length]
            loss_energy = loss_energy[:min_length]
            print(f"Truncated both to length: {min_length}")
        
        # Get actual number of features from model output
        actual_n_features = score_per_feature.shape[1] if len(score_per_feature.shape) > 1 else 1
        print(f"Actual number of features from model: {actual_n_features}")
        
        # Create feature names that match the actual model output
        if feature_names is not None:
            print(f"Feature names from CSV: {len(feature_names)} features")
            if len(feature_names) != actual_n_features:
                print(f"WARNING: Mismatch between CSV features ({len(feature_names)}) and model features ({actual_n_features})")
                # Use the minimum to avoid index errors
                min_features = min(len(feature_names), actual_n_features)
                if len(feature_names) > actual_n_features:
                    feature_names_used = feature_names[:actual_n_features]
                    print(f"Truncated CSV feature names to match model output: {feature_names_used}")
                else:
                    # Extend with generic names if CSV has fewer features
                    feature_names_used = feature_names + [f"Feature_{i}" for i in range(len(feature_names), actual_n_features)]
                    print(f"Extended feature names to match model output: {feature_names_used}")
            else:
                feature_names_used = feature_names
        else:
            feature_names_used = [f"Feature_{i}" for i in range(actual_n_features)]
            print(f"Created generic feature names: {feature_names_used}")
        
        # Adjust timestamps to match the timestep-level data (5200 timesteps)
        if test_dates is not None:
            test_data_length = len(test_dates)
            
            print(f"Test data length: {test_data_length}")
            print(f"Model output length: {len(loss_energy)}")
            
            # Create timestamps for each timestep
            if len(loss_energy) <= test_data_length:
                # Map each model output to corresponding test data timestamp
                # Assuming the model outputs correspond to consecutive timesteps in test data
                timestamps = test_dates[:len(loss_energy)]
            else:
                # If model output is longer than test data, extend timestamps
                base_timestamps = test_dates
                additional_needed = len(loss_energy) - len(test_dates)
                if additional_needed > 0:
                    freq = pd.infer_freq(test_dates) or '1H'
                    additional_timestamps = pd.date_range(
                        start=test_dates[-1] + pd.Timedelta(freq),
                        periods=additional_needed,
                        freq=freq
                    )
                    timestamps = base_timestamps.append(additional_timestamps)
                else:
                    timestamps = base_timestamps
        else:
            timestamps = None
        
        # Determine threshold
        if threshold is None:
            threshold = np.percentile(loss_energy, percentile_threshold)
            print(f"Auto-determined threshold ({percentile_threshold}th percentile): {threshold:.4f}")
        else:
            print(f"Using provided threshold: {threshold:.4f}")
        
        # Detect anomalies
        anomaly_mask = loss_energy > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = loss_energy[anomaly_indices]
        
        print(f"Detected {len(anomaly_indices)} anomalies out of {len(loss_energy)} samples")
        if len(anomaly_indices) > 0:
            print(f"Anomaly indices: {anomaly_indices}")
            print(f"Max anomaly index: {max(anomaly_indices)}")
            print(f"Score_per_feature available indices: 0 to {len(score_per_feature)-1}")
        print(f"Anomaly rate: {len(anomaly_indices)/len(loss_energy)*100:.2f}%")
        
        # Verify that all anomaly indices are within bounds
        valid_anomaly_indices = anomaly_indices[anomaly_indices < len(score_per_feature)]
        if len(valid_anomaly_indices) < len(anomaly_indices):
            print(f"WARNING: {len(anomaly_indices) - len(valid_anomaly_indices)} anomaly indices are out of bounds")
            print(f"Using only {len(valid_anomaly_indices)} valid anomaly indices")
            anomaly_indices = valid_anomaly_indices
            anomaly_scores = loss_energy[anomaly_indices]
        
        # Feature attribution for each anomaly
        anomaly_results = []
        
        for idx, anomaly_idx in enumerate(anomaly_indices):
            # Verify index is within bounds
            if anomaly_idx >= len(score_per_feature):
                print(f"Skipping anomaly at index {anomaly_idx} (out of bounds)")
                continue
                
            # Get feature scores for this anomaly timestep
            # score_per_feature is now (5200, 8) - one row per timestep
            feature_scores = score_per_feature[anomaly_idx]  # Shape: (8,) - feature contributions for this timestep
            
            # Feature scores are already per-feature contributions for this timestep
            mean_feature_contribution = feature_scores
            
            # Verify we have the right number of features
            if len(mean_feature_contribution) != len(feature_names_used):
                print(f"WARNING: Feature count mismatch at anomaly {anomaly_idx}")
                print(f"  Feature contributions: {len(mean_feature_contribution)}")
                print(f"  Feature names: {len(feature_names_used)}")
                # Take minimum to avoid index errors
                min_features = min(len(mean_feature_contribution), len(feature_names_used))
                mean_feature_contribution = mean_feature_contribution[:min_features]
                feature_names_used = feature_names_used[:min_features]
                print(f"  Truncated to {min_features} features")
            
            # Find top contributing features
            top_feature_indices = np.argsort(mean_feature_contribution)[::-1]
            
            # Get timestamp info
            if timestamps is not None:
                anomaly_timestamp = timestamps.iloc[anomaly_idx]
                # Extract time series window around anomaly
                start_idx = max(0, anomaly_idx - window_size)
                end_idx = min(len(timestamps), anomaly_idx + window_size + 1)
                window_timestamps = timestamps[start_idx:end_idx]
                window_scores = loss_energy[start_idx:end_idx]
                window_feature_scores = score_per_feature[start_idx:end_idx]
            else:
                anomaly_timestamp = anomaly_idx
                # Extract time series window around anomaly using indices
                start_idx = max(0, anomaly_idx - window_size)
                end_idx = min(len(loss_energy), anomaly_idx + window_size + 1)
                window_timestamps = list(range(start_idx, end_idx))
                window_scores = loss_energy[start_idx:end_idx]
                window_feature_scores = score_per_feature[start_idx:end_idx]
            
            anomaly_info = {
                'anomaly_index': anomaly_idx,
                'anomaly_timestamp': anomaly_timestamp,
                'anomaly_score': anomaly_scores[idx],
                'threshold': threshold,
                'top_contributing_features': [
                    {
                        'feature_name': feature_names_used[feat_idx],
                        'feature_index': feat_idx,
                        'contribution_score': mean_feature_contribution[feat_idx],
                        'rank': rank + 1
                    }
                    for rank, feat_idx in enumerate(top_feature_indices[:5])  # Top 5 features
                ],
                'all_feature_contributions': {
                    feature_names_used[i]: float(mean_feature_contribution[i])  # Ensure JSON serializable
                    for i in range(len(mean_feature_contribution))
                },
                'time_window': {
                    'start_idx': start_idx,
                    'end_idx': end_idx - 1,
                    'timestamps': window_timestamps,
                    'anomaly_scores': window_scores.tolist(),
                    'feature_scores': window_feature_scores.tolist() if hasattr(window_feature_scores, 'tolist') else window_feature_scores
                }
            }
            
            anomaly_results.append(anomaly_info)
        
        # Overall statistics
        results = {
            'threshold': threshold,
            'total_samples': len(loss_energy),
            'anomaly_count': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(loss_energy),
            'anomaly_indices': anomaly_indices.tolist(),
            'all_scores': loss_energy.tolist(),
            'anomaly_details': anomaly_results,
            'feature_names': feature_names_used,
            'timestamps': timestamps.tolist() if timestamps is not None else None,
            'test_split_info': {
                'test_split': test_split,
                'win_size': win_size,
                'step': step,
                'test_data_length': len(test_dates) if test_dates is not None else None,
                'full_data_length': len(all_dates) if all_dates is not None else None
            }
        }
        
        return results
    
    def print_anomaly_summary(self, results: Dict, show_all=False, max_anomalies=None):
        """Print a summary of detected anomalies"""
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Total samples: {results['total_samples']}")
        print(f"Anomalies detected: {results['anomaly_count']}")
        print(f"Anomaly rate: {results['anomaly_rate']*100:.2f}%")
        print(f"Threshold: {results['threshold']:.4f}")
        
        if results['anomaly_count'] > 0:
            sorted_anomalies = sorted(results['anomaly_details'], 
                                    key=lambda x: x['anomaly_score'], reverse=True)
            
            if show_all or max_anomalies is None:
                print(f"\nAll {len(sorted_anomalies)} anomalies by score:")
                anomalies_to_show = sorted_anomalies
            else:
                max_show = max_anomalies if max_anomalies else 5
                print(f"\nTop {min(max_show, len(sorted_anomalies))} anomalies by score:")
                anomalies_to_show = sorted_anomalies[:max_show]
            
            for i, anomaly in enumerate(anomalies_to_show):
                print(f"\n{i+1}. Timestamp: {anomaly['anomaly_timestamp']}")
                print(f"   Score: {anomaly['anomaly_score']:.4f}")
                if anomaly['top_contributing_features']:
                    print(f"   Top contributing feature: {anomaly['top_contributing_features'][0]['feature_name']}")
                    print(f"   Feature contribution: {anomaly['top_contributing_features'][0]['contribution_score']:.4f}")
                    
                    # Show top 3 contributing features for each anomaly
                    print(f"   Top 3 features:")
                    for j, feat in enumerate(anomaly['top_contributing_features'][:3]):
                        print(f"     {j+1}. {feat['feature_name']}: {feat['contribution_score']:.4f}")
    
    def print_all_anomalies(self, results: Dict):
        """Convenience method to print all anomalies"""
        self.print_anomaly_summary(results, show_all=True)
    
    def plot_anomalies(self, results: Dict, timestamps=None, max_plots=3):
        """Plot anomaly detection results"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if results['anomaly_count'] == 0:
            print("No anomalies to plot")
            return
        
        # Plot 1: Overall anomaly scores
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        x_axis = timestamps if timestamps is not None else range(len(results['all_scores']))
        plt.plot(x_axis, results['all_scores'], 'b-', alpha=0.7, label='Anomaly Score')
        plt.axhline(y=results['threshold'], color='r', linestyle='--', label='Threshold')
        
        # Mark anomalies
        anomaly_indices = results['anomaly_indices']
        if timestamps is not None:
            anomaly_x = [timestamps[i] for i in anomaly_indices]
        else:
            anomaly_x = anomaly_indices
        anomaly_y = [results['all_scores'][i] for i in anomaly_indices]
        plt.scatter(anomaly_x, anomaly_y, color='red', s=50, label='Anomalies', zorder=5)
        
        plt.title('Anomaly Detection Results')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Feature contributions for top anomalies
        plt.subplot(2, 1, 2)
        top_anomalies = sorted(results['anomaly_details'], 
                              key=lambda x: x['anomaly_score'], reverse=True)[:max_plots]
        
        if top_anomalies and top_anomalies[0]['top_contributing_features']:
            for i, anomaly in enumerate(top_anomalies):
                features = [f['feature_name'] for f in anomaly['top_contributing_features'][:5]]
                contributions = [f['contribution_score'] for f in anomaly['top_contributing_features'][:5]]
                
                x_pos = np.arange(len(features)) + i * 0.25
                plt.bar(x_pos, contributions, width=0.2, 
                       label=f"Anomaly {i+1} (t={anomaly['anomaly_timestamp']})")
            
            plt.title('Top Contributing Features for Major Anomalies')
            plt.xlabel('Features')
            plt.ylabel('Contribution Score')
            plt.legend()
            plt.xticks(np.arange(len(features)), features, rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results: Dict, filename: str = "anomaly_results.csv"):
        """Export anomaly results to CSV"""
        import pandas as pd
        
        if results['anomaly_count'] == 0:
            print("No anomalies to export")
            return
        
        # Create DataFrame with anomaly details
        export_data = []
        for anomaly in results['anomaly_details']:
            if not anomaly['top_contributing_features']:
                continue
                
            row = {
                'anomaly_index': anomaly['anomaly_index'],
                'anomaly_timestamp': anomaly['anomaly_timestamp'],
                'anomaly_score': anomaly['anomaly_score'],
                'top_feature': anomaly['top_contributing_features'][0]['feature_name'],
                'top_feature_score': anomaly['top_contributing_features'][0]['contribution_score']
            }
            
            # Add top 5 feature contributions
            for i, feat in enumerate(anomaly['top_contributing_features'][:5]):
                row[f'feature_{i+1}_name'] = feat['feature_name']
                row[f'feature_{i+1}_score'] = feat['contribution_score']
            
            export_data.append(row)
        
        if export_data:
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        else:
            print("No valid anomaly data to export")

# Usage example with your specific test_loader
def example_usage_with_your_data():
    """Example of how to use the anomaly detector with your specific data (last 10%)"""
    
    # Your test loader configuration (last 10% of data)
    test_loader = get_loader_segment(
        data_path='dataset/diy_pakem.csv', 
        batch_size=256, 
        win_size=100, 
        step=100,
        mode='test',  # This should handle the last 10% split
        dataset='all'
    )
    
    # Assuming you have your model instance
    # model_instance = YourModelClass(...)
    # detector = AnomalyDetector(model_instance)
    
    # Detect anomalies - the function will automatically extract feature names and dates
    # and properly align them with the test split (last 10%)
    # results = detector.detect_anomalies_with_attribution(
    #     test_loader=test_loader,
    #     data_path='dataset/diy_pakem.csv',  # Extract features & dates
    #     test_split=0.1,                     # Last 10% for testing
    #     win_size=100,                       # Match your loader config
    #     step=100,                           # Match your loader config  
    #     percentile_threshold=95,            # Use 95th percentile as threshold
    #     window_size=5                       # ±5 timesteps around anomaly
    # )
    
    # Print summary
    # detector.print_anomaly_summary(results)
    
    # Plot results
    # detector.plot_anomalies(results, results.get('timestamps'))
    
    # Export results
    # detector.export_results(results, "diy_pakem_anomaly_results.csv")
    
    print("Example setup complete. Uncomment the lines above to run.")
    pass

