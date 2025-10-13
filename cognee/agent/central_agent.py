"""
kafka_workforce.py

Kafka-driven multi-agent workforce system with Weather, Sensor, and Maintenance agents.
Listens to Kafka topic for anomaly events and processes them automatically.

Updated: Workforce is triggered by Kafka messages instead of user input.
- Sensor Agent → sensor_knowledge dataset only
- Maintenance Agent → maintenance_knowledge dataset only
"""

# Standard library imports
import sys
import os
import warnings
import logging
import json
import signal
from datetime import datetime
from pathlib import Path

# Third-party imports
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from dotenv import load_dotenv

# CAMEL framework imports
from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.tasks import Task

# Import your existing modules
from tools import *
from prompt_template import *

# Import direct Cognee tools with dataset binding
from cognee_direct_tools import get_cognee_tools, get_sensor_tools, get_maintenance_tools

# Suppress async warnings and verbose logs for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('camel').setLevel(logging.ERROR)
logging.getLogger('WorkforceLogger').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Suppress model output logs
os.environ['CAMEL_VERBOSE'] = 'false'


# Initialize local LLM model
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11440/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16384,
    },
)


def initialize_cognee():
    """Check Cognee is ready (does not prune existing data)."""
    print("\n[0/5] Checking Cognee status...")
    try:
        from cognee_direct_tools import initialize_cognee_sync
        # This no longer prunes data - just checks initialization
        initialize_cognee_sync()
        print("✓ Cognee ready (existing data preserved)")
    except Exception as e:
        print(f"⚠ Warning: Cognee check issue: {e}")
        print("✓ Continuing (Cognee will auto-initialize on first use)")


def setup_workforce():
    """
    Setup workforce with Weather, Sensor, and Maintenance agents.
    
    Returns:
        Workforce: Configured workforce ready for use
    """
    
    print("\n" + "="*70)
    print("INITIALIZING MULTI-AGENT WORKFORCE")
    print("="*70)
    
    # Initialize Cognee database first
    initialize_cognee()

    ##======================== Weather Agent ========================
    print("\n[1/5] Setting up Weather Agent...")
    weather_tool = FunctionTool(get_weather_param)
    
    weather_agent = ChatAgent(
        system_message=WEATHER_AGENT_PROMPT,
        tools=[weather_tool],
        model=ollama_model,
    )
    print("✓ Weather Agent ready")
    
    ##======================== Sensor Agent (sensor_knowledge dataset) ========================
    print("\n[2/5] Setting up Sensor Agent with Cognee Tools...")
    
    # Get sensor-specific tools (bound to sensor_knowledge dataset)
    sensor_tools = get_cognee_tools(
        context_name="sensor",
        dataset_name="sensor_knowledge",
        include_prune=True
    )
    
    sensor_agent = ChatAgent(
        system_message=SENSOR_AGENT_PROMPT,
        tools=sensor_tools,
        model=ollama_model,
    )
    print("✓ Sensor Agent ready (bound to sensor_knowledge dataset)")
    
    ##======================== Maintenance Agent (maintenance_knowledge dataset) ========================
    print("\n[3/5] Setting up Maintenance Agent with Cognee Tools...")

    # Get maintenance-specific tools (bound to maintenance_knowledge dataset)
    maintenance_tools = get_cognee_tools(
        context_name="maintenance",
        dataset_name="maintenance_knowledge",
        include_prune=True
    )
    
    maintenance_agent = ChatAgent(
        system_message=MAINTENANCE_AGENT_PROMPT,
        tools=maintenance_tools,
        model=ollama_model,
    )
    print("✓ Maintenance Agent ready (bound to maintenance_knowledge dataset)")
    
    ##======================== Task Agent ========================
    print("\n[4/5] Setting up Task Agent...")
    
    task_agent = ChatAgent(
        system_message=TASK_AGENT_PROMPT,
        model=ollama_model,
    )
    print("✓ Task Agent ready")
    
    ##======================== Coordinator Agent ========================
    print("\n[5/5] Setting up Coordinator Agent...")
    
    coordinator_agent = ChatAgent(
        system_message=COORDINATOR_AGENT_PROMPT,
        model=ollama_model,
    )
    print("✓ Coordinator Agent ready")
    
    ##======================== Build Workforce ========================
    print("\n" + "="*70)
    print("Building Workforce...")
    print("="*70)
    
    workforce = Workforce(
        description='Workforce for analyzing the anomaly of weather sensor data and provide report to the user',
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
        graceful_shutdown_timeout=15.0,
        share_memory=False,
        use_structured_output_handler=True,
    )
    
    # Add worker agents
    workforce.add_single_agent_worker(
        worker=weather_agent,
        description='Retrieves and analyzes weather data including temperature, precipitation, wind, and other meteorological parameters'
    ).add_single_agent_worker(
        worker=sensor_agent,
        description='Searches and manages sensor specifications and technical documentation'
    ).add_single_agent_worker(
        worker=maintenance_agent,
        description='Searches and manages maintenance related information, maintenance logs, repair histories, equipment status, and service records'
    )

    print("\n✓ Workforce ready:")
    print("   - Weather Analyst: Weather data analysis")
    print("   - Sensor Monitor: Sensor specs (sensor_knowledge dataset only)")
    print("   - Maintenance Expert: Maintenance logs (maintenance_knowledge dataset only)")
    print("\n✓ Dataset isolation enforced:")
    print("   - Sensor Agent → sensor_knowledge only")
    print("   - Maintenance Agent → maintenance_knowledge only")
    print("="*70)
    print()
    
    return workforce


class EventMonitor:
    """Monitor Kafka events and trigger workforce for each anomaly"""
    
    def __init__(self, 
                 workforce: Workforce,
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "weather-anomalies",
                 group_id: str = "workforce-consumer-group",
                 log_dir: str = "logs/anomaly_analysis"):
        """
        Initialize Kafka event monitor.
        
        Args:
            workforce: Initialized workforce to process events
            bootstrap_servers: Kafka broker address
            topic: Kafka topic to monitor
            group_id: Consumer group ID
            log_dir: Directory to store anomaly analysis logs
        """
        self.workforce = workforce
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.running = False
        self.processed_count = 0
        
        # Setup logging directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session log file
        self.session_start = datetime.now()
        self.session_log_file = self.log_dir / f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Initialize session log
        self._init_session_log()
    
    def _init_session_log(self):
        """Initialize session log file."""
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("KAFKA WORKFORCE SESSION LOG\n")
            f.write("="*70 + "\n")
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Kafka Broker: {self.bootstrap_servers}\n")
            f.write(f"Topic: {self.topic}\n")
            f.write(f"Consumer Group: {self.group_id}\n")
            f.write(f"Log Directory: {self.log_dir.absolute()}\n")
            f.write("="*70 + "\n\n")
        
        print(f"\n📁 Session log: {self.session_log_file}")
    
    def _log_to_session(self, message: str):
        """Append message to session log."""
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    
    def _save_anomaly_log(self, anomaly_data: dict, query: str, result: str, processing_time: float):
        """
        Save detailed log for each anomaly processed.
        
        Args:
            anomaly_data: Original anomaly data from Kafka
            query: Formatted query sent to workforce
            result: Workforce analysis result
            processing_time: Time taken to process (seconds)
        """
        # Create filename with timestamp and station name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        station = anomaly_data.get('station_name', 'Unknown').replace(' ', '_')
        filename = f"anomaly_{timestamp}_{station}.log"
        log_file = self.log_dir / filename
        
        # Write detailed log
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ANOMALY ANALYSIS LOG\n")
            f.write("="*70 + "\n\n")
            
            # Processing metadata
            f.write("PROCESSING METADATA\n")
            f.write("-"*70 + "\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"Processed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Anomaly Count: #{self.processed_count}\n")
            f.write(f"Log File: {filename}\n")
            f.write("\n")
            
            # Original anomaly data
            f.write("ORIGINAL ANOMALY DATA\n")
            f.write("-"*70 + "\n")
            f.write(json.dumps(anomaly_data, indent=2))
            f.write("\n\n")
            
            # Formatted query
            f.write("WORKFORCE QUERY\n")
            f.write("-"*70 + "\n")
            f.write(query)
            f.write("\n\n")
            
            # Workforce analysis result
            f.write("WORKFORCE ANALYSIS RESULT\n")
            f.write("-"*70 + "\n")
            f.write(result if result else "No analysis generated")
            f.write("\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF LOG\n")
            f.write("="*70 + "\n")
        
        return log_file
        
    def connect(self):
        """Connect to Kafka broker."""
        try:
            print(f"\n🔌 Connecting to Kafka broker: {self.bootstrap_servers}")
            print(f"📡 Subscribing to topic: {self.topic}")
            
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='latest',  # Start from latest messages
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000  # Timeout for checking running flag
            )
            
            print("✓ Connected to Kafka successfully")
            return True
            
        except KafkaError as e:
            print(f"❌ Kafka connection error: {e}")
            return False
    
    def format_anomaly_query(self, anomaly_data: dict) -> str:
        """
        Format anomaly JSON into a natural language query for the workforce.
        
        Args:
            anomaly_data: Anomaly JSON data from Kafka
            
        Returns:
            Formatted query string
        """
        timestamp = anomaly_data.get('timestamp', 'unknown time')
        features = anomaly_data.get('anomalous_features', {})
        station = anomaly_data.get('station_name', 'Unknown Station')
        location = anomaly_data.get('location', {})
        analyses = anomaly_data.get('analysis', {})  # Get analysis for each parameter
        
        # Build feature list with analysis
        feature_descriptions = []
        feature_map = {
            'tt': 'Temperature',
            'rh': 'Relative Humidity',
            'pp': 'Precipitation',
            'ws': 'Wind Speed',
            'wd': 'Wind Direction',
            'sr': 'Solar Radiation',
            'rr': 'Rainfall Rate'
        }
        
        for code, value in features.items():
            name = feature_map.get(code, code)
            analysis = analyses.get(code, 'No analysis available')
            feature_descriptions.append(
                f"{name} ({code}): {value}\n  Analysis: {analysis}"
            )
        
        # Construct comprehensive query
        query = f"""Analyze the following weather anomaly detected at {station}:

Time: {timestamp}
Location: Latitude {location.get('latitude', 'N/A')}, Longitude {location.get('longitude', 'N/A')}

Anomalous measurements with initial analysis:
{chr(10).join('- ' + desc for desc in feature_descriptions)}

Please investigate further:
1. What are the current weather conditions at this location and how do they compare?
2. Are these sensor readings within the operational range of the equipment?
3. Check if there's any recent maintenance history that might explain this anomaly
4. Consider the initial analysis provided and provide a comprehensive assessment with recommendations"""
        
        return query
    
    def process_anomaly(self, anomaly_data: dict):
        """
        Process a single anomaly event through the workforce.
        
        Args:
            anomaly_data: Anomaly JSON data from Kafka
        """
        start_time = datetime.now()
        
        try:
            timestamp = anomaly_data.get('timestamp', 'unknown')
            station = anomaly_data.get('station_name', 'Unknown')
            
            print("\n" + "="*70)
            print(f"🚨 NEW ANOMALY DETECTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            print(f"Station: {station}")
            print(f"Timestamp: {timestamp}")
            print(f"Anomalous Features: {list(anomaly_data.get('anomalous_features', {}).keys())}")
            print("="*70)
            
            # Log to session
            self._log_to_session(f"New anomaly from {station} - Features: {list(anomaly_data.get('anomalous_features', {}).keys())}")
            
            # Format query from anomaly data
            query = self.format_anomaly_query(anomaly_data)
            
            print("\n🔄 Activating workforce...\n")
            
            # Process through workforce
            import io
            import contextlib
            
            # Capture stdout to suppress verbose output
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = self.workforce.process_task(Task(content=query))
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Get result text
            result_text = result.result if result.result else "No analysis generated"
            
            # Display result
            print("\n✓ WORKFORCE ANALYSIS COMPLETE:")
            print("="*70)
            print(result_text)
            print("="*70)
            
            self.processed_count += 1
            
            # Save detailed log
            log_file = self._save_anomaly_log(anomaly_data, query, result_text, processing_time)
            
            print(f"\n📊 Total anomalies processed: {self.processed_count}")
            print(f"📁 Log saved: {log_file.name}")
            
            # Log to session
            self._log_to_session(f"Anomaly #{self.processed_count} processed in {processing_time:.2f}s - Log: {log_file.name}")
            
        except Exception as e:
            error_msg = f"Error processing anomaly: {e}"
            print(f"\n❌ {error_msg}")
            self._log_to_session(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Save error log
            try:
                error_result = f"ERROR: {str(e)}\n\n{traceback.format_exc()}"
                processing_time = (datetime.now() - start_time).total_seconds()
                self._save_anomaly_log(
                    anomaly_data, 
                    query if 'query' in locals() else "Query generation failed",
                    error_result,
                    processing_time
                )
            except:
                pass
    
    def start_monitoring(self):
        """Start monitoring Kafka topic for anomaly events."""
        if not self.connect():
            print("❌ Failed to connect to Kafka. Exiting.")
            return
        
        self.running = True
        
        print("\n" + "="*70)
        print("🎯 WORKFORCE MONITORING ACTIVE")
        print("="*70)
        print(f"Waiting for anomaly events on topic '{self.topic}'...")
        print("Press Ctrl+C to stop")
        print("="*70)
        
        try:
            while self.running:
                try:
                    # Poll for messages
                    for message in self.consumer:
                        if not self.running:
                            break
                        
                        # Process the anomaly
                        anomaly_data = message.value
                        self.process_anomaly(anomaly_data)
                        
                except StopIteration:
                    # Timeout occurred, continue loop
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n⏸️  Stopping monitoring...")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup."""
        self.running = False
        if self.consumer:
            self.consumer.close()
            print("✓ Kafka consumer closed")
        
        # Write session summary
        session_end = datetime.now()
        session_duration = (session_end - self.session_start).total_seconds()
        
        summary = f"""
{'='*70}
SESSION SUMMARY
{'='*70}
Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
Session End: {session_end.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {session_duration:.2f} seconds ({session_duration/60:.2f} minutes)
Total Anomalies Processed: {self.processed_count}
{'='*70}
"""
        
        print(summary)
        
        # Append to session log
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + summary)
        
        print(f"📁 Full session log: {self.session_log_file}")
        print(f"📁 Individual logs: {self.log_dir.absolute()}")
        print("\n👋 Monitoring stopped")


def main():
    """Main execution."""
    
    try:
        print("\n" + "="*70)
        print("STARTING KAFKA-DRIVEN WORKFORCE SYSTEM")
        print("Dataset-Isolated Multi-Agent Architecture")
        print("="*70)
        
        # Setup workforce
        workforce = setup_workforce()
        
        # Initialize Kafka event monitor
        monitor = EventMonitor(
            workforce=workforce,
            bootstrap_servers="localhost:9092",  # Adjust as needed
            topic="weather-anomalies",           # Adjust as needed
            group_id="workforce-consumer-group",
            log_dir="logs/anomaly_analysis"      # Adjust as needed
        )
        
        # Start monitoring for anomaly events
        monitor.start_monitoring()
        
        print("\n" + "="*70)
        print("SESSION ENDED")
        print("="*70)
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)