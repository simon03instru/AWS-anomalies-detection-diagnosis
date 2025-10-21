# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Model checkpoints and data
checkpoints/
dataset/
*.pth
*.csv

# Logs and output
logs/
output/
*.log
*.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

python evaluate.py \
  --dataset /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/dataset/dataset.csv \
  --test-data /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/dataset/synthetic_data_with_anomalies.csv \
  --checkpoint /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/checkpoints/all_checkpoint.pth \
  --threshold 0.3 \
  --apply-adjustment \
  --apply-lag \
  --lag-tolerance 10


python test_generator.py \
  --dataset /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/dataset/dataset.csv \
  --output synthetic_data_with_anomalies.csv \
  --n-anomalies 90 \
  --min-duration 1 \
  --max-duration 3 \
  --min-gap 20\
  --seed 42

  python multiple_threshold_evaluation.py \
  --dataset /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/dataset/dataset.csv \
  --test-data /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/dataset/synthetic_data_with_anomalies.csv \
  --checkpoint /home/ubuntu/running/anomaly_detection/anomaly_monitor_diy_stageof_yogyakarta/checkpoints/all_checkpoint.pth \
  --thresholds 0.7,0.8,0.9,1.0,1.1 \
  --apply-adjustment \
  --apply-lag \
  --lag-tolerance 10