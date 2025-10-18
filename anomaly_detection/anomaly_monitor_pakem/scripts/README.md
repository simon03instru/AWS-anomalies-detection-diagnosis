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
  --dataset /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/dataset/dataset.csv \
  --test-data /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/dataset/synthetic_data_with_anomalies.csv \
  --checkpoint /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/checkpoints/all_checkpoint.pth \
  --threshold 1.0 \
  --apply-adjustment \
  --apply-lag \
  --lag-tolerance 1

python evaluate.py \
  --dataset /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/dataset/dataset.csv \
  --test-data /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/dataset/synthetic_data_with_anomalies.csv \
  --checkpoint /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/checkpoints/all_checkpoint.pth \
  --threshold 1.0 \
  --exclude-latency \
  --latency-window 5

python test_generator.py \
  --dataset /home/ubuntu/running/anomaly_detection/anomaly_monitor_dki_taman_mini/dataset/dataset.csv \
  --output synthetic_data_with_anomalies.csv \
  --n-anomalies 60 \
  --min-duration 1 \
  --max-duration 3 \
  --min-gap 20\
  --seed 42