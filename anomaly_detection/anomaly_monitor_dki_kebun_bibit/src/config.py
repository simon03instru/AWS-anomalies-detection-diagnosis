# config.py - Updated version with fixed threshold
import os

class Config:
    """Configuration settings for the weather anomaly monitoring system"""
    
    STORAGE_DB_PATH = "data/weather_data.db"
    STORAGE_CSV_PATH = "data/weather_data.csv"
    
    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Model paths
    MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'all_checkpoint.pth')
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset.csv')
    
    # Scaler settings
    USE_ROBUST_SCALER = True
    #SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'scaler', 'diy_pakem_scaler.pkl')
    
    # Output paths
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    
    # MQTT Settings
    MQTT_BROKER = "10.33.205.40"
    MQTT_PORT = 1883
    MQTT_TOPIC = "station/aws_dki_kebun_bibit"
    MQTT_QOS = 1
    MQTT_CLIENT_ID = "weather_anomaly_monitor_dki_kebun_bibit"

    # Model configuration
    MODEL_CONFIG = {
        'data_path': DATASET_PATH,
        'batch_size': 32,
        'win_size': 100,
        'step': 100,
        'input_c': 7,
        'output_c': 7,
        'dataset': 'all',
        'lr': 0.001,
        'num_epochs': 10,
        'k': 0.1,
        'model_save_path': os.path.join(BASE_DIR, 'checkpoints', 'diy_pakem'),
        'anomaly_ratio': 0.1
    }
    
    # Weather features - adjust to match your data format
    WEATHER_FEATURES = ['tt', 'rh', 'pp', 'ws', 'wd', 'sr', 'rr']
    
    # Detection settings
    BUFFER_SIZE = 1000
    WIN_SIZE = 100
    PREFILL_DUMMY_DATA = True
    
    # FIXED THRESHOLD SETTINGS
    USE_FIXED_THRESHOLD = True
    FIXED_THRESHOLD = 1.0
    
    # Backup settings
    ANOMALY_THRESHOLD_PERCENTILE = 95.0
    TEMPERATURE_SCALE = 5
    
    # Output settings
    OUTPUT_FORMAT = "json"
    LOG_LEVEL = "INFO"
    SHOW_DETAILED_DEBUG = True
    SHOW_FEATURE_SCORES = True
    
    # Create directories if they don't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)