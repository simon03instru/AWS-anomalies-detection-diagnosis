"""
Configuration loader for the application
"""

import os
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPICS: List[str] = os.getenv("KAFKA_TOPICS", "").split(",")
    
    # Agent Configuration
    REPORTING_AGENT_ENABLED: bool = os.getenv("REPORTING_AGENT_ENABLED", "true").lower() == "true"
    DIAGNOSIS_AGENT_ENABLED: bool = os.getenv("DIAGNOSIS_AGENT_ENABLED", "true").lower() == "true"
    ALERTING_AGENT_ENABLED: bool = os.getenv("ALERTING_AGENT_ENABLED", "true").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # System Monitor Configuration
    MONITOR_INTERVAL_SECONDS: int = int(os.getenv("MONITOR_INTERVAL_SECONDS", "60"))
    MONITOR_SERVER_ID: str = os.getenv("MONITOR_SERVER_ID", "prod-server-01")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        if not cls.KAFKA_BOOTSTRAP_SERVERS:
            errors.append("KAFKA_BOOTSTRAP_SERVERS is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True


# Validate configuration on import
try:
    Config.validate()
    print("✅ Configuration loaded successfully")
except ValueError as e:
    print(f"⚠️  Configuration warning: {e}")
