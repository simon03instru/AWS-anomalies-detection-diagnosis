"""
# CAMEL AI Event-Driven Multi-Agent System

Event-driven architecture for intelligent agents using CAMEL AI and Apache Kafka.

## Features

- 🤖 **CAMEL AI Integration**: Intelligent agents powered by LLMs
- 📡 **Event-Driven Architecture**: Asynchronous agent communication via Kafka
- 🔄 **Loose Coupling**: Agents communicate through events, not direct calls
- 📊 **Real-time Monitoring**: System metrics analysis and reporting
- 🔍 **Intelligent Diagnosis**: AI-powered root cause analysis
- 🚨 **Smart Alerting**: Context-aware alert generation

## Prerequisites

- Python 3.8 or higher
- Apache Kafka (running on localhost:9092)
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/camel-event-driven-agents.git
cd camel-event-driven-agents
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Kafka Setup

### Option 1: Using Docker (Recommended)

```bash
docker-compose up -d
```

### Option 2: Manual Setup

1. Download Kafka from https://kafka.apache.org/downloads
2. Start Zookeeper:
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

3. Start Kafka:
```bash
bin/kafka-server-start.sh config/server.properties
```

## Usage

Run the demo:
```bash
python camel_agents.py
```

## Architecture

```
SystemMonitor → [SYSTEM_METRICS_COLLECTED] → ReportingAgent (CAMEL)
                                                    ↓
                                          [SYSTEM_REPORT_GENERATED]
                                                    ↓
                                             DiagnosisAgent (CAMEL)
                                                    ↓
                                          [DIAGNOSIS_COMPLETED]
                                                    ↓
                                              AlertingAgent
                                                    ↓
                                              [ALERT_SENT]
```

## Event Topics

- `system-metrics`: Raw system metrics and logs
- `system-reports`: Analyzed reports from ReportingAgent
- `diagnosis-results`: Diagnosis from DiagnosisAgent
- `alerts`: Alert notifications

## Configuration

See `.env.example` for all configuration options.

Key settings:
- `OPENAI_API_KEY`: Your OpenAI API key
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka server address
- `OPENAI_MODEL`: LLM model to use (default: gpt-4o-mini)

## Development

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
"""