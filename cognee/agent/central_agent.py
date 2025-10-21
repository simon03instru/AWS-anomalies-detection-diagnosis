"""
kafka_workforce.py with Comprehensive Logging and New JSON Format Support

Kafka-driven multi-agent workforce system with detailed logging of:
- Agent interactions and decision-making
- Tool calls with full arguments and results
- Message exchanges between agents
- Final output synthesis
- Complete execution flow
- DETAILED ERROR TRACKING AND DEBUGGING

Updated to support WEATHER_ANOMALY_CONFIRMED event format
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

# Configure logging - ENABLE DEBUG MODE FOR CAMEL
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# ENABLE CAMEL DEBUG LOGGING TO SEE INTERNAL ERRORS
logging.basicConfig(level=logging.INFO)
logging.getLogger('camel').setLevel(logging.DEBUG)
logging.getLogger('camel.societies.workforce').setLevel(logging.DEBUG)
logging.getLogger('WorkforceLogger').setLevel(logging.DEBUG)

load_dotenv()
os.environ['CAMEL_VERBOSE'] = 'false'


# ============================================================================
# Comprehensive Logging System with Error Tracking
# ============================================================================

class WorkforceLogger:
    """Detailed logger for multi-agent interactions with enhanced error tracking"""
    
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
        
        # Timing
        self.agent_start_time = None
        self.task_start_time = None
        
        # Error tracking
        self.error_count = 0
        self.errors = []
    
    def start_anomaly_log(self, anomaly_id: str, anomaly_data: dict) -> Path:
        """Start a new detailed log file for an anomaly"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Extract station name from new format
        station = anomaly_data.get('data', {}).get('station_id', 'Unknown').replace(' ', '_')
        filename = f"detailed_{timestamp}_{station}.log"
        
        self.current_log_file = self.log_dir / filename
        self.current_log_handle = open(self.current_log_file, 'w', encoding='utf-8')
        
        # Reset counters
        self.agent_call_count = {}
        self.tool_call_count = {}
        self.current_depth = 0
        self.error_count = 0
        self.errors = []
        
        # Write header
        self._write_section("DETAILED ANOMALY PROCESSING LOG")
        self._write(f"Anomaly ID: {anomaly_id}")
        self._write(f"Event Type: {anomaly_data.get('event_type', 'N/A')}")
        self._write(f"Correlation ID: {anomaly_data.get('correlation_id', 'N/A')}")
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
            self._write(f"Total Errors: {self.error_count}")
            
            self._write(f"\nAgent Call Statistics:")
            for agent, count in self.agent_call_count.items():
                self._write(f"  - {agent}: {count} calls")
            
            self._write(f"\nTool Call Statistics:")
            for tool, count in self.tool_call_count.items():
                self._write(f"  - {tool}: {count} calls")
            
            if self.errors:
                self._write(f"\n⚠️  ERRORS ENCOUNTERED ({len(self.errors)}):")
                for idx, error in enumerate(self.errors, 1):
                    self._write(f"\n  Error #{idx}:")
                    self._write(f"    Type: {error['type']}")
                    self._write(f"    Message: {error['message']}")
                    self._write(f"    Location: {error['location']}")
                    self._write(f"    Time: {error['time']}")
            
            self._write_separator()
            self._write("END OF LOG")
            self._write("="*80)
            
            self.current_log_handle.close()
            self.current_log_handle = None
    
    def log_workforce_start(self, query: str):
        """Log the start of workforce processing"""
        self.task_start_time = datetime.now()
        self._write_section("WORKFORCE PROCESSING STARTED")
        self._write("Initial Query:")
        self._write(query)
        self._write_separator()
    
    def log_agent_call(self, agent_name: str, input_msg: str, call_number: int):
        """Log when an agent is called"""
        self.agent_call_count[agent_name] = self.agent_call_count.get(agent_name, 0) + 1
        self.agent_start_time = datetime.now()
        
        self._write_section(f"AGENT CALL #{call_number}: {agent_name}")
        self._write(f"Agent: {agent_name}")
        self._write(f"Call Count: {self.agent_call_count[agent_name]}")
        self._write(f"Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        self._write(f"\nInput Message:")
        self._write(self._indent(input_msg, 2))
    
    def log_agent_response(self, agent_name: str, output_msg: str, terminated: bool):
        """Log agent response with timing"""
        elapsed = 0
        if self.agent_start_time:
            elapsed = (datetime.now() - self.agent_start_time).total_seconds()
        
        self._write(f"\nOutput Message:")
        self._write(self._indent(output_msg, 2))
        self._write(f"\nTerminated: {terminated}")
        self._write(f"Elapsed Time: {elapsed:.2f}s")
        
        if elapsed > 30:
            self._write(f"⚠️  WARNING: Slow response ({elapsed:.2f}s)")
        
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
        import traceback
        
        self.error_count += 1
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'location': f'Tool: {tool_name}',
            'time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_info)
        
        self._write(f"\n❌ ERROR in {tool_name}:")
        self._write(self._indent(f"Type: {type(error).__name__}", 2))
        self._write(self._indent(f"Message: {str(error)}", 2))
        self._write(f"\nTraceback:")
        self._write(self._indent(traceback.format_exc(), 2))
        self._write_separator(char="-")
    
    def log_agent_error(self, agent_name: str, error: Exception, input_msg: str = None):
        """Log agent execution error with full context"""
        import traceback
        
        self.error_count += 1
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'location': f'Agent: {agent_name}',
            'time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_info)
        
        self._write_section(f"❌ AGENT FAILURE: {agent_name}", char="!")
        self._write(f"Error Type: {type(error).__name__}")
        self._write(f"Error Message: {str(error)}")
        
        if input_msg:
            self._write(f"\nInput Message (first 500 chars):")
            self._write(self._indent(input_msg[:500], 2))
            if len(input_msg) > 500:
                self._write(f"  ... (truncated {len(input_msg) - 500} characters)")
        
        self._write(f"\nFull Traceback:")
        self._write(self._indent(traceback.format_exc(), 2))
        self._write_separator(char="!")
    
    def log_workforce_error(self, error: Exception, context: str = None):
        """Log workforce-level error"""
        import traceback
        
        self.error_count += 1
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'location': f'Workforce: {context}' if context else 'Workforce',
            'time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_info)
        
        self._write_section("❌❌❌ WORKFORCE TASK FAILURE ❌❌❌", char="!")
        if context:
            self._write(f"Context: {context}")
        self._write(f"Error Type: {type(error).__name__}")
        self._write(f"Error Message: {str(error)}")
        self._write(f"\nFull Traceback:")
        self._write(self._indent(traceback.format_exc(), 2))
        self._write_section("❌❌❌ END FAILURE LOG ❌❌❌", char="!")
    
    def log_stdout_capture(self, output: str):
        """Log captured stdout from workforce"""
        if output and output.strip():
            self._write_section("STDOUT CAPTURE", char="-")
            self._write(output)
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
# Logged Agent Wrapper with Enhanced Error Handling
# ============================================================================

class LoggedChatAgent(ChatAgent):
    """ChatAgent wrapper with comprehensive logging and error tracking"""
    
    def __init__(self, *args, agent_name: str = "Agent", **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = agent_name
        self.call_count = 0
    
    def step(self, input_message: BaseMessage, *args, **kwargs):
        """Logged step method with error tracking"""
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
            # Log the error with full context
            workforce_logger.log_agent_error(self.agent_name, e, input_content)
            
            # Also print to console for immediate visibility
            print(f"\n❌ {self.agent_name} failed: {type(e).__name__}: {e}")
            
            # Re-raise to let workforce handle it
            raise


# ============================================================================
# Logged Tool Wrapper
# ============================================================================

def log_tool(tool_func: Callable, tool_name: str = None) -> Callable:
    """Decorator to log tool function calls with error handling"""
    
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
            
            # Print to console for immediate visibility
            print(f"\n❌ Tool {actual_tool_name} failed: {type(e).__name__}: {e}")
            
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
    print("   - All errors captured with stack traces")
    print("   - Decision trees captured")
    print("   - Final output synthesis logged")
    print("="*70)
    print()
    
    return workforce


# ============================================================================
# Event Monitor with Enhanced Error Handling - UPDATED FOR NEW FORMAT
# ============================================================================

class EventMonitor:
    """Monitor Kafka events with comprehensive logging and error tracking"""
    
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
        self.failed_count = 0
        
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
    
    def format_anomaly_query(self, event_data: dict) -> str:
        """
        Format query from new WEATHER_ANOMALY_CONFIRMED event structure
        
        Expected structure:
        {
          "event_type": "WEATHER_ANOMALY_CONFIRMED",
          "timestamp": "2025-10-15T13:07:26.090310",
          "source": "StationAgent-AWS_DIY_STAGEOF_YOGYAKARTA",
          "data": {
            "timestamp": "2025-10-14T04:42:36.239425",
            "station_id": "AWS_DIY_STAGEOF_YOGYAKARTA",
            "station_metadata": {...},
            "confirmed_anomalies": [
              {
                "parameter": "Temperature",
                "value": -999,
                "sensor_brand": "Vaisala HMP155",
                "trend_analysis": "..."
              }
            ],
            "validation_timestamp": "2025-10-15T13:07:26.090287",
            "has_trend_analysis": true
          },
          "correlation_id": "anomaly-AWS_DIY_STAGEOF_YOGYAKARTA-1760533646"
        }
        """
        
        # Extract data section
        data = event_data.get('data', {})
        
        # Basic info
        event_timestamp = data.get('timestamp', 'unknown')
        station_id = data.get('station_id', 'Unknown')
        
        # Station metadata
        metadata = data.get('station_metadata', {})
        location = metadata.get('location', 'Unknown')
        latitude = metadata.get('latitude', 'N/A')
        longitude = metadata.get('longitude', 'N/A')
        altitude = metadata.get('altitude', 'N/A')
        
        # Confirmed anomalies
        confirmed_anomalies = data.get('confirmed_anomalies', [])
        
        # Build anomaly descriptions
        anomaly_descriptions = []
        for idx, anomaly in enumerate(confirmed_anomalies, 1):
            parameter = anomaly.get('parameter', 'Unknown')
            value = anomaly.get('value', 'N/A')
            sensor_brand = anomaly.get('sensor_brand', 'N/A')
            trend_analysis = anomaly.get('trend_analysis', 'No analysis available')
            
            anomaly_desc = f"""Anomaly #{idx}: {parameter}
  Value: {value}
  Sensor Brand: {sensor_brand}
  Trend Analysis: {trend_analysis}"""
            
            anomaly_descriptions.append(anomaly_desc)
        
        # Build complete query
        query = f"""Analyze the following weather anomaly detected at {station_id}:

Event Detection Time: {event_timestamp}
Station: {station_id}
Location: {location}
Coordinates: Latitude {latitude}, Longitude {longitude}, Altitude {altitude}m

{len(confirmed_anomalies)} Confirmed Anomal{"y" if len(confirmed_anomalies) == 1 else "ies"} with Trend Analysis:
{chr(10).join(anomaly_descriptions)}

Validation Timestamp: {data.get('validation_timestamp', 'N/A')}
Correlation ID: {event_data.get('correlation_id', 'N/A')}

Please investigate further:
1. What are the current weather conditions at this location and how do they compare to the anomalous readings?
2. Are these sensor readings within the operational range of the equipment? Is the sensor sensitivity matched to the observed values?
3. Check if there's any troubleshooting information that might explain these anomalies or sensor output and what maintenance actions should be recommended.
4. Consider the trend analysis provided and provide a comprehensive assessment with actionable recommendations.
5. If multiple anomalies are present, analyze potential correlations or common causes."""
        
        return query
    
    def extract_anomaly_summary(self, event_data: dict) -> dict:
        """Extract summary information for logging"""
        data = event_data.get('data', {})
        confirmed_anomalies = data.get('confirmed_anomalies', [])
        
        return {
            'station_id': data.get('station_id', 'Unknown'),
            'event_type': event_data.get('event_type', 'Unknown'),
            'timestamp': data.get('timestamp', 'Unknown'),
            'anomaly_count': len(confirmed_anomalies),
            'parameters': [a.get('parameter', 'Unknown') for a in confirmed_anomalies],
            'correlation_id': event_data.get('correlation_id', 'N/A')
        }
    
    def process_anomaly(self, event_data: dict):
        """Process anomaly event in new format"""
        global workforce_logger
        start_time = datetime.now()
        
        try:
            # Validate event type
            event_type = event_data.get('event_type')
            if event_type != 'WEATHER_ANOMALY_CONFIRMED':
                print(f"\n⚠️  Skipping non-anomaly event: {event_type}")
                return
            
            # Extract summary info
            summary = self.extract_anomaly_summary(event_data)
            station_id = summary['station_id']
            parameters = summary['parameters']
            anomaly_count = summary['anomaly_count']
            
            # Create anomaly ID
            data = event_data.get('data', {})
            timestamp = data.get('timestamp', 'unknown')
            anomaly_id = f"{station_id}_{timestamp}".replace(' ', '_').replace(':', '-')
            
            print("\n" + "="*70)
            print(f"🚨 ANOMALY DETECTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            print(f"Station: {station_id}")
            print(f"Event Timestamp: {timestamp}")
            print(f"Anomaly Count: {anomaly_count}")
            print(f"Parameters: {', '.join(parameters)}")
            print(f"Correlation ID: {summary['correlation_id']}")
            print("="*70)
            
            # Start detailed log
            log_file = workforce_logger.start_anomaly_log(anomaly_id, event_data)
            
            self._log_to_session(f"New anomaly from {station_id} - {anomaly_count} parameter(s): {', '.join(parameters)}")
            
            # Format query
            query = self.format_anomaly_query(event_data)
            
            # Log workforce start
            workforce_logger.log_workforce_start(query)
            
            print("\n🔄 Activating workforce (all interactions will be logged)...\n")
            
            # Process through workforce with comprehensive error handling
            result_text = None
            
            try:
                # Capture stdout for debugging
                import io, contextlib
                f = io.StringIO()
                
                with contextlib.redirect_stdout(f):
                    result = self.workforce.process_task(Task(content=query))
                
                # Log captured stdout
                stdout_output = f.getvalue()
                if stdout_output and stdout_output.strip():
                    workforce_logger.log_stdout_capture(stdout_output)
                    print("\n📝 Captured workforce stdout:")
                    print(stdout_output)
                
                # Get result
                result_text = result.result if result.result else "No result generated"
                
            except Exception as task_error:
                # LOG THE ACTUAL ERROR WITH FULL DETAILS
                workforce_logger.log_workforce_error(task_error, "process_task")
                
                print(f"\n❌❌❌ WORKFORCE TASK FAILED ❌❌❌")
                print(f"Error Type: {type(task_error).__name__}")
                print(f"Error Message: {str(task_error)}")
                print(f"📁 Full error details logged to: {log_file.name}")
                
                # Print stack trace to console
                import traceback
                print("\nStack Trace:")
                traceback.print_exc()
                
                # Set result to error message
                result_text = f"ERROR: Task processing failed - {type(task_error).__name__}: {str(task_error)}"
                
                # Increment failed count
                self.failed_count += 1
                self._log_to_session(f"FAILED: {station_id} - {type(task_error).__name__}: {str(task_error)}")
            
            # Log final output (even if it's an error message)
            output_summary = {
                'length': len(result_text),
                'word_count': len(result_text.split()),
                'line_count': len(result_text.split('\n')),
                'is_error': result_text.startswith('ERROR:')
            }
            workforce_logger.log_final_output(result_text, output_summary)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # End log
            workforce_logger.end_anomaly_log(processing_time)
            
            print("\n✓ ANALYSIS COMPLETE:")
            print("="*70)
            print(result_text)
            print("="*70)
            
            if not result_text.startswith('ERROR:'):
                self.processed_count += 1
                self._log_to_session(f"SUCCESS: {station_id} - {processing_time:.2f}s - {anomaly_count} anomalies")
            
            print(f"\n📊 Statistics:")
            print(f"   Successful: {self.processed_count}")
            print(f"   Failed: {self.failed_count}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"📁 Detailed log: {log_file.name}")
            print(f"   → Check this file for full error details and stack traces")
            
        except Exception as e:
            # Catch any errors in the processing logic itself
            print(f"\n❌ Unexpected error in process_anomaly: {e}")
            import traceback
            traceback.print_exc()
            
            if workforce_logger and workforce_logger.current_log_handle:
                workforce_logger._write("\n\n❌❌❌ UNEXPECTED ERROR IN PROCESSING LOGIC ❌❌❌")
                workforce_logger._write(f"Error: {e}")
                workforce_logger._write(traceback.format_exc())
                
                processing_time = (datetime.now() - start_time).total_seconds()
                workforce_logger.end_anomaly_log(processing_time)
            
            self.failed_count += 1
            self._log_to_session(f"UNEXPECTED ERROR: {e}")
    
    def start_monitoring(self):
        if not self.connect():
            return
        
        self.running = True
        
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE LOGGING MONITORING ACTIVE")
        print("="*70)
        print(f"Topic: '{self.topic}'")
        print(f"Expected Event Type: WEATHER_ANOMALY_CONFIRMED")
        print(f"Logs: {self.log_dir.absolute()}")
        print(f"Debug Mode: ENABLED (CAMEL internal logs visible)")
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
        
        print(f"\n✓ Session complete:")
        print(f"   Successful: {self.processed_count}")
        print(f"   Failed: {self.failed_count}")
        print(f"   Total: {self.processed_count + self.failed_count}")
        print(f"📁 Logs: {self.log_dir.absolute()}")


def main():
    try:
        print("\n" + "="*70)
        print("KAFKA WORKFORCE WITH COMPREHENSIVE ERROR LOGGING")
        print("UPDATED FOR WEATHER_ANOMALY_CONFIRMED EVENT FORMAT")
        print("="*70)
        print("✓ CAMEL debug logging ENABLED")
        print("✓ Full error tracking and stack traces")
        print("✓ Tool call monitoring")
        print("✓ Agent interaction logging")
        print("✓ Support for multiple anomalies per event")
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
        print(f"\n❌ Fatal error: {e}")
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


