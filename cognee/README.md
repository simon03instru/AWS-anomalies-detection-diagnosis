# Central Diagnosis Agent

Knowledge-based diagnostic system for multi-station weather anomaly analysis using CAMEL AI and Cognee RAG (Retrieval-Augmented Generation).

---

## 🎯 Overview

The Central Diagnosis Agent:
- **Event Broker** from all weather stations via Kafka
- **Compare with current actual weather data** acsess data from Open Meteo API
- **Diagnoses root causes** using RAG-based reasoning
- **Provides maintenance recommendations** from knowledge base
- **Manages knowledge base** through interactive interface


## 📁 Folder Structure
```
cognee/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── .env.example                       # Environment template
├── .env                               # Your configuration (gitignored)
│
├── agent/
│   ├── central_agent.py               # Main agent implementation
│   ├── prompt_template.py             # System prompts & templates
│   ├── tools.py                       # Weather data retrieval tools
│   ├── cognee_direct_tools.py         # Knowledge base access tools
│   │
│   └── document/                      # Knowledge Base Documents
│       ├── maintenance_knowledge/
│       │   ├── pressure_maintenance.txt
│       │   ├── temperature_sensor_guide.txt
│       │   ├── humidity_calibration.txt
│       │   ├── wind_sensor_maintenance.txt
│       │   └── general_troubleshooting.txt
│       │
│       └── sensor_specifications/
│           ├── ....
│
├── interactive_workforce.py           # Knowledge base management UI
├── logs/
    ├── detailed_event.log

```

---

## 🚀 Quick Start

### 1. Create Virtual Environment
```bash
cd cognee
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
```txt
# AI Framework
camel-ai[all]==0.2.1

# Knowledge Base
cognee>=0.1.0

# Communication
kafka-python==2.0.2
paho-mqtt>=1.6.0

# Utilities
python-dotenv==1.0.1
colorlog==6.8.2

# Data Processing
pandas>=1.3.0
numpy>=1.21.0
```

### 3. Configure Environment

Create `.env` file:
```bash
cp .env.example .env
nano .env
```

**`.env` Configuration**:
```bash
# ==================== LLM Configuration ====================
LLM_API_KEY=sk-your-openai-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7

# ==================== Cognee Configuration ====================
ENABLE_BACKEND_ACCESS_CONTROL=true
REQUIRE_AUTHENTICATION=true


### 4. Run Central Agent
```bash
python agent/central_agent.py
```


## 📚 Knowledge Base Management

### Interactive Interface

Launch the knowledge management UI:
```bash
python interactive_workforce.py
```

**Interface Example**:
```
================================================================================
COGNEE KNOWLEDGE BASE - INTERACTIVE MANAGEMENT
================================================================================

Commands:
  - insert <node> <file_path>  : Add document to knowledge base
  - query <question>           : Test knowledge retrieval
  - list                       : Show all indexed documents
  - delete <node>              : Remove document from knowledge base
  - help                       : Show this help message
  - exit                       : Exit interface

================================================================================

You: Insert this document to maintenance agent knowledge base with node "pressure-sensor": "/home/ubuntu/running/cognee/agent/document/maintenance_knowledge/pressure_maintenance.txt"

Agent: Processing document insertion...
       ✓ Document loaded: pressure_maintenance.txt (1,245 words)
       ✓ Vectorized and indexed under node: "pressure-sensor"
       ✓ Knowledge base updated successfully

       Document Summary:
       - Type: Maintenance Procedure
       - Sensor: Pressure (Piezoelectric)
       - Topics: Calibration, fault diagnosis, replacement
       - Related nodes: sensor-calibration, troubleshooting

You: query What are common pressure sensor faults?

Agent: Retrieving relevant knowledge...
       
       Based on knowledge base:
       
       Common pressure sensor faults include:
       
       1. **Drift**: Gradual deviation from true readings
          - Cause: Sensor aging, temperature effects
          - Detection: Comparison with reference standard
          - Solution: Recalibration or replacement
       
       2. **Offset Error**: Constant bias in readings
          - Cause: Improper zero-point calibration
          - Detection: Check readings at known pressure
          - Solution: Zero-point recalibration
       
       3. **Noise/Erratic Readings**: Random fluctuations
          - Cause: Electrical interference, mechanical vibration
          - Detection: Statistical analysis of variance
          - Solution: Check connections, shield cables
       
       4. **Complete Failure**: No output signal
          - Cause: Power supply failure, sensor damage
          - Detection: Check power and signal continuity
          - Solution: Replace sensor
       
       [Source: pressure_maintenance.txt, sensor-troubleshooting.txt]

You: list

Agent: Indexed Documents:
       
       Node: pressure-sensor
         - pressure_maintenance.txt (1,245 words)
         - piezoelectric_pressure_specs.txt (856 words)
       
       Node: temperature-sensor
         - temperature_sensor_guide.txt (1,532 words)
         - pt100_rtd_specs.txt (1,024 words)
       
       Node: humidity-sensor
         - humidity_calibration.txt (980 words)
         - capacitive_humidity_specs.txt (742 words)
       
       Node: wind-sensor
         - wind_sensor_maintenance.txt (1,155 words)
         - anemometer_specs.txt (895 words)
       
       Node: general
         - general_troubleshooting.txt (2,100 words)
       
       Total: 12 documents, 11,529 words

You: exit

Agent: Saving knowledge base state...
       ✓ Knowledge base saved
       Goodbye!

================================================================================
```

### Document Organization

Store documents in organized folders:
```
agent/document/
├── maintenance_knowledge/          # How to fix things
│   ├── pressure_maintenance.txt
│   ├── temperature_sensor_guide.txt
│   ├── humidity_calibration.txt
│   ├── wind_sensor_maintenance.txt
│   └── general_troubleshooting.txt
│
└── sensor_specifications/          # Technical specs
    ├── pt100_rtd_specs.txt
    ├── capacitive_humidity_specs.txt
    ├── piezoelectric_pressure_specs.txt
    └── anemometer_specs.txt
```

### Knowledge Base Best Practices

1. **Structure Documents Clearly**
```
   # Pressure Sensor Maintenance Guide
   
   ## Overview
   Brief description of sensor type and common issues
   
   ## Calibration Procedure
   Step-by-step calibration instructions
   
   ## Common Faults
   - Fault type 1: Description, causes, solutions
   - Fault type 2: Description, causes, solutions
   
   ## Replacement Procedure
   Steps for sensor replacement
   
   ## Related Documents
   Links to specifications, troubleshooting guides
```

2. **Use Descriptive Node Names**
```python
   # Good
   "pressure-sensor"
   "temperature-calibration"
   "humidity-troubleshooting"
   
   # Avoid
   "doc1"
   "misc"
   "temp"
```

3. **Keep Documents Updated**
```bash
   # Update existing document
   python interactive_workforce.py
   > delete pressure-sensor
   > insert pressure-sensor /path/to/updated_pressure_maintenance.txt
```

---
