"""
kafka_workforce.py with Comprehensive Logging

Kafka-driven multi-agent workforce system with detailed logging of:
- Agent interactions and decision-making
- Tool calls with full arguments and results
- Message exchanges between agents
- Final output synthesis
- Complete execution flow
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
from functools import wraps
from typing import Any, Callable, Dict, List

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
from camel.messages import BaseMessage

# Import your existing modules
from tools import *
from prompt_template import *
from cognee_direct_tools import get_cognee_tools

# Suppress async warnings and verbose logs
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('camel').setLevel(logging.ERROR)
logging.getLogger('WorkforceLogger').setLevel(logging.ERROR)

load_dotenv()
os.environ['CAMEL_VERBOSE'] = 'false'


# ============================================================================
# Comprehensive Logging System
# ============================================================================

class WorkforceLogger:
    """Detailed logger for multi-agent interactions"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current anomaly log file
        self.current_log_file = None
        self.current_log_handle = None
        
        # Counters
        self.agent_call_count = {}
        self.tool_call_count = {}
        self.current_depth = 0
    
    def start_anomaly_log(self, anomaly_id: str, anomaly_data: dict) -> Path:
        """Start a new detailed log file for an anomaly"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        station = anomaly_data.get('station_name', 'Unknown').replace(' ', '_')
        filename = f"detailed_{timestamp}_{station}.log"
        
        self.current_log_file = self.log_dir / filename
        self.current_log_handle = open(self.current_log_file, 'w', encoding='utf-8')
        
        # Reset counters
        self.agent_call_count = {}
        self.tool_call_count = {}
        self.current_depth = 0
        
        # Write header
        self._write_section("DETAILED ANOMALY PROCESSING LOG")
        self._write(f"Anomaly ID: {anomaly_id}")
        self._write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"Log File: {filename}")
        self._write_separator()
        
        self._write_section("ORIGINAL ANOMALY DATA")
        self._write(json.dumps(anomaly_data, indent=2))
        self._write_separator()
        
        return self.current_log_file
    
    def end_anomaly_log(self, processing_time: float):
        """Close the current anomaly log"""
        if self.current_log_handle:
            self._write_separator()
            self._write_section("PROCESSING SUMMARY")
            self._write(f"Total Processing Time: {processing_time:.2f} seconds")
            self._write(f"\nAgent Call Statistics:")
            for agent, count in self.agent_call_count.items():
                self._write(f"  - {agent}: {count} calls")
            
            self._write(f"\nTool Call Statistics:")
            for tool, count in self.tool_call_count.items():
                self._write(f"  - {tool}: {count} calls")
            
            self._write_separator()
            self._write("END OF LOG")
            self._write("="*80)
            
            self.current_log_handle.close()
            self.current_log_handle = None
    
    def log_workforce_start(self, query: str):
        """Log the start of workforce processing"""
        self._write_section("WORKFORCE PROCESSING STARTED")
        self._write("Initial Query:")
        self._write(query)
        self._write_separator()
    
    def log_agent_call(self, agent_name: str, input_msg: str, call_number: int):
        """Log when an agent is called"""
        self.agent_call_count[agent_name] = self.agent_call_count.get(agent_name, 0) + 1
        
        self._write_section(f"AGENT CALL #{call_number}: {agent_name}")
        self._write(f"Agent: {agent_name}")
        self._write(f"Call Count: {self.agent_call_count[agent_name]}")
        self._write(f"Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        self._write(f"\nInput Message:")
        self._write(self._indent(input_msg, 2))
    
    def log_agent_response(self, agent_name: str, output_msg: str, terminated: bool):
        """Log agent response"""
        self._write(f"\nOutput Message:")
        self._write(self._indent(output_msg, 2))
        self._write(f"\nTerminated: {terminated}")
        self._write_separator()
    
    def log_tool_call(self, tool_name: str, args: tuple, kwargs: dict, call_number: int):
        """Log when a tool is called"""
        self.tool_call_count[tool_name] = self.tool_call_count.get(tool_name, 0) + 1
        
        self._write_section(f"TOOL CALL #{call_number}: {tool_name}", char="-")
        self._write(f"Tool: {tool_name}")
        self._write(f"Call Count: {self.tool_call_count[tool_name]}")
        self._write(f"Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        if args:
            self._write(f"\nArguments:")
            self._write(self._indent(str(args), 2))
        
        if kwargs:
            self._write(f"\nKeyword Arguments:")
            self._write(self._indent(json.dumps(kwargs, indent=2, default=str), 2))
    
    def log_tool_result(self, tool_name: str, result: Any, execution_time: float):
        """Log tool execution result"""
        result_str = str(result)
        result_len = len(result_str)
        
        self._write(f"\nExecution Time: {execution_time:.3f} seconds")
        self._write(f"Result Length: {result_len} characters")
        self._write(f"\nResult:")
        
        # Format result nicely if it's JSON-like
        try:
            if isinstance(result, dict):
                self._write(self._indent(json.dumps(result, indent=2, default=str), 2))
            elif result_len > 5000:
                self._write(self._indent(result_str[:2000], 2))
                self._write(f"\n  ... (truncated {result_len - 2000} characters) ...")
                self._write(f"\n  [Full result length: {result_len} characters]")
            else:
                self._write(self._indent(result_str, 2))
        except:
            self._write(self._indent(result_str[:2000] if result_len > 2000 else result_str, 2))
        
        self._write_separator(char="-")
    
    def log_tool_error(self, tool_name: str, error: Exception):
        """Log tool execution error"""
        self._write(f"\n❌ ERROR in {tool_name}:")
        self._write(self._indent(str(error), 2))
        self._write_separator(char="-")
    
    def log_decision_point(self, decision_maker: str, decision: str, reasoning: str = None):
        """Log decision points in the workflow"""
        self._write_section(f"DECISION POINT: {decision_maker}", char="~")
        self._write(f"Decision: {decision}")
        if reasoning:
            self._write(f"Reasoning:")
            self._write(self._indent(reasoning, 2))
        self._write_separator(char="~")
    
    def log_final_output(self, output: str, summary: dict = None):
        """Log the final synthesized output"""
        self._write_section("FINAL OUTPUT - SYNTHESIZED RESULT")
        self._write(output)
        
        if summary:
            self._write(f"\n\nOutput Statistics:")
            for key, value in summary.items():
                self._write(f"  - {key}: {value}")
        
        self._write_separator()
    
    def log_message_exchange(self, from_agent: str, to_agent: str, message: str):
        """Log inter-agent message exchanges"""
        self._write_section(f"MESSAGE: {from_agent} → {to_agent}", char="·")
        self._write(self._indent(message, 2))
        self._write_separator(char="·")
    
    def _write(self, text: str):
        """Write to current log file"""
        if self.current_log_handle:
            self.current_log_handle.write(text + "\n")
            self.current_log_handle.flush()
    
    def _write_section(self, title: str, char: str = "="):
        """Write a section header"""
        self._write("")
        self._write(char * 80)
        self._write(title)
        self._write(char * 80)
    
    def _write_separator(self, char: str = "="):
        """Write a separator line"""
        self._write(char * 80)
        self._write("")
    
    def _indent(self, text: str, levels: int = 1) -> str:
        """Indent text by specified levels"""
        indent = "  " * levels
        return "\n".join(indent + line for line in text.split("\n"))


# Global logger instance
workforce_logger = None


# ============================================================================
# Logged Agent Wrapper
# ============================================================================

class LoggedChatAgent(ChatAgent):
    """ChatAgent wrapper with comprehensive logging"""
    
    def __init__(self, *args, agent_name: str = "Agent", **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = agent_name
        self.call_count = 0
    
    def step(self, input_message: BaseMessage, *args, **kwargs):
        """Logged step method"""
        self.call_count += 1
        global workforce_logger
        
        # Log agent call
        input_content = str(input_message.content)
        workforce_logger.log_agent_call(self.agent_name, input_content, self.call_count)
        
        start_time = datetime.now()
        
        try:
            # Call original step method
            result = super().step(input_message, *args, **kwargs)
            
            # Log response
            if result and hasattr(result, 'msg') and result.msg:
                output_content = str(result.msg.content)
                terminated = getattr(result, 'terminated', False)
                workforce_logger.log_agent_response(
                    self.agent_name, 
                    output_content, 
                    terminated
                )
            
            return result
            
        except Exception as e:
            workforce_logger._write(f"\n❌ ERROR in {self.agent_name}: {str(e)}")
            workforce_logger._write_separator()
            raise


# ============================================================================
# Logged Tool Wrapper
# ============================================================================

def log_tool(tool_func: Callable, tool_name: str = None) -> Callable:
    """Decorator to log tool function calls"""
    
    actual_tool_name = tool_name or tool_func.__name__
    tool_call_counter = {'count': 0}
    
    @wraps(tool_func)
    def logged_wrapper(*args, **kwargs):
        global workforce_logger
        
        tool_call_counter['count'] += 1
        call_num = tool_call_counter['count']
        
        # Log tool call
        workforce_logger.log_tool_call(actual_tool_name, args, kwargs, call_num)
        
        start_time = datetime.now()
        
        try:
            # Execute tool
            result = tool_func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log result
            workforce_logger.log_tool_result(actual_tool_name, result, execution_time)
            
            return result
            
        except Exception as e:
            workforce_logger.log_tool_error(actual_tool_name, e)
            raise
    
    return logged_wrapper


# ============================================================================
# Create Logged Tools
# ============================================================================

def create_logged_tools(tools_list: list, context_name: str = "tool") -> list:
    """Wrap tools with logging"""
    logged_tools = []
    
    for tool in tools_list:
        if isinstance(tool, FunctionTool):
            original_func = tool.func
            tool_name = getattr(tool, 'name', original_func.__name__)
            
            # Wrap with logging
            logged_func = log_tool(original_func, f"{context_name}_{tool_name}")
            
            # Create new FunctionTool
            logged_tool = FunctionTool(logged_func)
            for attr in ['name', 'description']:
                if hasattr(tool, attr):
                    setattr(logged_tool, attr, getattr(tool, attr))
            
            logged_tools.append(logged_tool)
        else:
            logged_tools.append(tool)
    
    return logged_tools


# ============================================================================
# Model Setup
# ============================================================================

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
    """Check Cognee is ready"""
    print("\n[0/5] Checking Cognee status...")
    try:
        from cognee_direct_tools import initialize_cognee_sync
        initialize_cognee_sync()
        print("✓ Cognee ready (existing data preserved)")
    except Exception as e:
        print(f"⚠ Warning: Cognee check issue: {e}")
        print("✓ Continuing (Cognee will auto-initialize on first use)")


def setup_workforce():
    """Setup workforce with logged agents"""
    
    print("\n" + "="*70)
    print("INITIALIZING MULTI-AGENT WORKFORCE WITH LOGGING")
    print("="*70)
    
    initialize_cognee()

    ##======================== Weather Agent ========================
    print("\n[1/5] Setting up Weather Agent...")
    weather_tool_func = log_tool(get_weather_param, "get_weather")
    weather_tool = FunctionTool(weather_tool_func)
    
    weather_agent = LoggedChatAgent(
        system_message=WEATHER_AGENT_PROMPT,
        tools=[weather_tool],
        model=ollama_model,
        agent_name="WeatherAgent"
    )
    print("✓ Weather Agent ready with logging")
    
    ##======================== Sensor Agent ========================
    print("\n[2/5] Setting up Sensor Agent...")
    sensor_tools_raw = get_cognee_tools(
        context_name="sensor",
        dataset_name="sensor_knowledge",
        include_prune=True
    )
    sensor_tools = create_logged_tools(sensor_tools_raw, "sensor")
    
    sensor_agent = LoggedChatAgent(
        system_message=SENSOR_AGENT_PROMPT,
        tools=sensor_tools,
        model=ollama_model,
        agent_name="SensorAgent"
    )
    print(f"✓ Sensor Agent ready with {len(sensor_tools)} logged tools")
    
    ##======================== Maintenance Agent ========================
    print("\n[3/5] Setting up Maintenance Agent...")
    maintenance_tools_raw = get_cognee_tools(
        context_name="maintenance",
        dataset_name="maintenance_knowledge",
        include_prune=True
    )
    maintenance_tools = create_logged_tools(maintenance_tools_raw, "maintenance")
    
    maintenance_agent = LoggedChatAgent(
        system_message=MAINTENANCE_AGENT_PROMPT,
        tools=maintenance_tools,
        model=ollama_model,
        agent_name="MaintenanceAgent"
    )
    print(f"✓ Maintenance Agent ready with {len(maintenance_tools)} logged tools")
    
    ##======================== Task Agent ========================
    print("\n[4/5] Setting up Task Agent...")
    task_agent = LoggedChatAgent(
        system_message=TASK_AGENT_PROMPT,
        model=ollama_model,
        agent_name="TaskAgent"
    )
    print("✓ Task Agent ready with logging")
    
    ##======================== Coordinator Agent ========================
    print("\n[5/5] Setting up Coordinator Agent...")
    coordinator_agent = LoggedChatAgent(
        system_message=COORDINATOR_AGENT_PROMPT,
        model=ollama_model,
        agent_name="CoordinatorAgent"
    )
    print("✓ Coordinator Agent ready with logging")
    
    ##======================== Build Workforce ========================
    print("\n" + "="*70)
    print("Building Workforce...")
    print("="*70)
    
    workforce = Workforce(
        description='Workforce for analyzing the anomaly of weather sensor data',
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
        graceful_shutdown_timeout=15.0,
        share_memory=False,
        use_structured_output_handler=True,
    )
    
    workforce.add_single_agent_worker(
        worker=weather_agent,
        description='Weather data analysis'
    ).add_single_agent_worker(
        worker=sensor_agent,
        description='Sensor specifications'
    ).add_single_agent_worker(
        worker=maintenance_agent,
        description='Maintenance information'
    )

    print("\n✓ Workforce ready with comprehensive logging:")
    print("   - All agent interactions logged")
    print("   - All tool calls logged with full details")
    print("   - Decision trees captured")
    print("   - Final output synthesis logged")
    print("="*70)
    print()
    
    return workforce


# ============================================================================
# Event Monitor
# ============================================================================

class EventMonitor:
    """Monitor Kafka events with comprehensive logging"""
    
    def __init__(self, 
                 workforce: Workforce,
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "weather-anomalies",
                 group_id: str = "workforce-consumer-group",
                 log_dir: str = "logs/anomaly_analysis"):
        
        self.workforce = workforce
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.running = False
        self.processed_count = 0
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        global workforce_logger
        workforce_logger = WorkforceLogger(self.log_dir)
        
        # Session log
        self.session_start = datetime.now()
        self.session_log_file = self.log_dir / f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.log"
        self._init_session_log()
    
    def _init_session_log(self):
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("KAFKA WORKFORCE SESSION LOG\n")
            f.write("="*70 + "\n")
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Kafka Broker: {self.bootstrap_servers}\n")
            f.write(f"Topic: {self.topic}\n")
            f.write(f"Log Directory: {self.log_dir.absolute()}\n")
            f.write("="*70 + "\n\n")
        
        print(f"\n📁 Session log: {self.session_log_file}")
    
    def _log_to_session(self, message: str):
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    
    def connect(self):
        try:
            print(f"\n🔌 Connecting to Kafka: {self.bootstrap_servers}")
            print(f"📡 Topic: {self.topic}")
            
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000
            )
            
            print("✓ Connected to Kafka")
            return True
            
        except KafkaError as e:
            print(f"❌ Kafka error: {e}")
            return False
    
    def format_anomaly_query(self, anomaly_data: dict) -> str:
        timestamp = anomaly_data.get('timestamp', 'unknown')
        features = anomaly_data.get('anomalous_features', {})
        station = anomaly_data.get('station_name', 'Unknown')
        location = anomaly_data.get('location', {})
        analyses = anomaly_data.get('analysis', {})
        
        feature_descriptions = []
        feature_map = {
            'tt': 'Temperature', 'rh': 'Relative Humidity', 'pp': 'Precipitation',
            'ws': 'Wind Speed', 'wd': 'Wind Direction', 'sr': 'Solar Radiation',
            'rr': 'Rainfall Rate'
        }
        
        for code, value in features.items():
            name = feature_map.get(code, code)
            analysis = analyses.get(code, 'No analysis available')
            feature_descriptions.append(
                f"{name} ({code}): {value}\n  Analysis: {analysis}"
            )
        
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
        global workforce_logger
        start_time = datetime.now()
        
        try:
            timestamp = anomaly_data.get('timestamp', 'unknown')
            station = anomaly_data.get('station_name', 'Unknown')
            features = list(anomaly_data.get('anomalous_features', {}).keys())
            
            anomaly_id = f"{station}_{timestamp}".replace(' ', '_').replace(':', '-')
            
            print("\n" + "="*70)
            print(f"🚨 ANOMALY DETECTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            print(f"Station: {station}")
            print(f"Timestamp: {timestamp}")
            print(f"Features: {features}")
            print("="*70)
            
            # Start detailed log
            log_file = workforce_logger.start_anomaly_log(anomaly_id, anomaly_data)
            
            self._log_to_session(f"New anomaly from {station} - Features: {features}")
            
            # Format query
            query = self.format_anomaly_query(anomaly_data)
            
            # Log workforce start
            workforce_logger.log_workforce_start(query)
            
            print("\n🔄 Activating workforce (all interactions will be logged)...\n")
            
            # Process through workforce
            import io, contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = self.workforce.process_task(Task(content=query))
            
            result_text = result.result if result.result else "No result"
            
            # Log final output
            output_summary = {
                'length': len(result_text),
                'word_count': len(result_text.split()),
                'line_count': len(result_text.split('\n'))
            }
            workforce_logger.log_final_output(result_text, output_summary)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # End log
            workforce_logger.end_anomaly_log(processing_time)
            
            print("\n✓ ANALYSIS COMPLETE:")
            print("="*70)
            print(result_text)
            print("="*70)
            
            self.processed_count += 1
            
            print(f"\n📊 Processed: {self.processed_count}")
            print(f"📁 Detailed log: {log_file.name}")
            print(f"   → All agent calls, tool calls, and decisions logged")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            if workforce_logger and workforce_logger.current_log_handle:
                processing_time = (datetime.now() - start_time).total_seconds()
                workforce_logger.end_anomaly_log(processing_time)
    
    def start_monitoring(self):
        if not self.connect():
            return
        
        self.running = True
        
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE LOGGING MONITORING ACTIVE")
        print("="*70)
        print(f"Topic: '{self.topic}'")
        print(f"Logs: {self.log_dir.absolute()}")
        print("Press Ctrl+C to stop")
        print("="*70)
        
        try:
            while self.running:
                try:
                    for message in self.consumer:
                        if not self.running:
                            break
                        self.process_anomaly(message.value)
                except StopIteration:
                    continue
        except KeyboardInterrupt:
            print("\n\n⏸️  Stopping...")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        self.running = False
        if self.consumer:
            self.consumer.close()
        
        print(f"\n✓ Session complete: {self.processed_count} anomalies processed")
        print(f"📁 Logs: {self.log_dir.absolute()}")


def main():
    try:
        print("\n" + "="*70)
        print("KAFKA WORKFORCE WITH COMPREHENSIVE LOGGING")
        print("="*70)
        
        # Setup workforce
        workforce = setup_workforce()
        
        # Start monitoring
        monitor = EventMonitor(
            workforce=workforce,
            bootstrap_servers="localhost:9092",
            topic="weather-anomalies",
            group_id="workforce-consumer-group",
            log_dir="logs/anomaly_analysis"
        )
        
        monitor.start_monitoring()
        
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