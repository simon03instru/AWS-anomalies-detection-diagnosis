# Station-Level Anomaly Detection

Independent weather station monitoring with Anomaly Transformer models and CAMEL AI agents for local anomaly detection and preliminary diagnosis.

---

## 🎯 Overview

Each weather station operates as an autonomous agent that:
- Monitors real-time sensor data (temperature, humidity, pressure, wind speed)
- Detects anomalies using trained Anomaly Transformer models
- Analyzes anomalies using local CAMEL AI agent
- Publishes events to central diagnosis system via kafka event broker

---

## 📁 Station Folder Structure
```
anomaly_monitor_xxx_xxx_xxx/
├── README.md                          # Station-specific guide
├── checkpoints/
│   └── all_checkpoint.pth             # Trained Anomaly Transformer model
├── data/
│   └── weather_data.db                # SQLite database (sliding window)
├── dataset/
│   ├── synthetic_test_anomalies.csv             # Synthetic anomalies for testing
│   └── dataset.csv              # Original data 
├── logs/
│   ├── anomaly_scores.csv             # Detection results
│   └── agent_thought_process.log      # Agent reasoning logs
├── src/
│   ├── config.py                      # Configuration file
│   ├── anomaly_detector.py            # Anomaly Transformer wrapper
│   ├── data_processor.py              # Data preprocessing
│   ├── mqtt_client.py                 # MQTT communication
│   ├── evaluate.py                    # Evaluate detection performance (result will be exported in the same folder)
│   └── ....                           # Other dependencies
├── station_agent.py                   # CAMEL AI agent implementation
├── weather_anomaly_mcp_server.py      # Station metadata & tools
├── main.py                            # Entry point
└── requirements.txt                   # Dependencies
```

---

## 🚀 Quick Start

### 1. Create Virtual Environment

**Important**: Use separate virtual environment for each station to enable independent deployment.
```bash
cd anomaly_monitor_xxx_xxx_xxx # or any station folder
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Station

Edit `src/config.py`:
```python

# ==================== Configuration ====================


# MQTT Configuration
# MQTT Settings
MQTT_BROKER = "10.33.205.40" # Your mqtt broker
MQTT_PORT = 1883
MQTT_TOPIC = "station/aws_xxx_xxx_xxx"
MQTT_QOS = 1
MQTT_CLIENT_ID = "weather_anomaly_monitor_xxx_xxx_xxx"


# Model Configuration
WINDOW_SIZE = 100                    # Sliding window size
ANOMALY_THRESHOLD = 0.7              # Detection threshold (0-1)
WEATHER_FEATURES = ['tt', 'rh', 'pp', 'ws', 'wd', 'sr', 'rr']


### 4. Configure Station Metadata

Edit `weather_anomaly_mcp_server.py` to define station-specific information and tools

### 5. Run Station Agent
```bash
python main.py
```

### 4. Add LLM API Key
Create .env and add your LLM API, DB_PATH, and KAFKA_Configuration:
```bash
DB_PATH=data/weather_data.db
OPENAI_API_KEY=""
GEMINI_API_KEY=""

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=10.33.205.40:9093 # change to event broker
KAFKA_TOPIC=weather-anomalies-test # Change to event topic

```

The anomaly detection will wait for incoming data from mqtt broker and will conduct online inference