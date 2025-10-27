#!/usr/bin/env python3
"""
Weather Anomaly Investigation Agent using CAMEL AI Platform with MCP Tools
Analyzes weather data for anomalies using MCP server tools via MCPToolkit.
Clean version with minimal debug output - logs to file only.
ENHANCED: Now logs complete analysis results to file.
"""

import sys
import json
import asyncio
import argparse
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.mcp_toolkit import MCPToolkit

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

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

class ThoughtProcessLogger:
    """Silent logger that only writes to file"""
    
    def __init__(self, log_file: str):
        self.log_entries = []
        self.log_file = log_file
        
        # Setup file logger
        self.file_logger = logging.getLogger('thought_process')
        self.file_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.file_logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.file_logger.addHandler(fh)
    
    def log_step(self, step_type: str, content: str):
        """Log a step silently to file only"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "step_type": step_type,
            "content": content
        }
        self.log_entries.append(entry)
        self.file_logger.info(f"[{step_type}] {content}")
    
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Log tool call to file"""
        args_str = json.dumps(arguments, indent=2)
        content = f"Calling tool: {tool_name}\nArguments:\n{args_str}"
        self.log_step("TOOL_CALL", content)

    def log_tool_result(self, tool_name: str, result: str, truncate: int = 200):
        """Log tool result to file - skip detailed logging for get_data_from_db"""
        # Skip detailed logging for database queries (too verbose)
        if tool_name == "get_data_from_db":
            # Just log a summary for database queries
            try:
                result_data = json.loads(result) if isinstance(result, str) else result
                record_count = len(result_data.get("records", [])) if isinstance(result_data, dict) else 0
                result_preview = f"[Database query returned {record_count} records, {len(result)} characters total]"
            except:
                result_preview = f"[Database query executed - {len(result)} characters returned]"
        else:
            # For other tools, show a preview
            result_preview = result[:truncate] + "..." if len(result) > truncate else result
        
        content = f"Tool '{tool_name}' returned: {result_preview}"
        self.log_step("TOOL_RESULT", content)

    
    def log_reasoning(self, reasoning: str):
        """Log agent's reasoning to file"""
        self.log_step("REASONING", reasoning)
    
    def log_decision(self, decision: str):
        """Log agent's decision to file"""
        self.log_step("DECISION", decision)
    
    def log_error(self, error: str):
        """Log an error to file"""
        self.log_step("ERROR", error)
    
    def log_analysis_start(self, anomaly_data: Dict[str, Any]):
        """Log the start of analysis to file"""
        content = f"Starting analysis of anomaly data:\n{json.dumps(anomaly_data, indent=2)}"
        self.log_step("ANALYSIS_START", content)
    
    def log_analysis_end(self, result: str):
        """Log the COMPLETE analysis result to file"""
        # Log the full result, not just the length
        separator = "=" * 80
        content = f"Analysis complete.\n\n{separator}\nFULL ANALYSIS RESULT:\n{separator}\n\n{result}\n\n{separator}"
        self.log_step("ANALYSIS_RESULT", content)
        
        # Also log a summary for quick reference
        summary = f"Result length: {len(result)} characters, {len(result.split())} words"
        self.log_step("ANALYSIS_SUMMARY", summary)
    
    def save_summary(self, output_file: str):
        """Save a summary of the thought process to a JSON file"""
        summary = {
            "session_start": self.log_entries[0]["timestamp"] if self.log_entries else None,
            "session_end": self.log_entries[-1]["timestamp"] if self.log_entries else None,
            "total_steps": len(self.log_entries),
            "steps": self.log_entries
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)


class WeatherAnomalyAgent:
    """Weather Anomaly Investigation Agent using CAMEL AI with MCP server tools"""
    
    def __init__(
        self, 
        mcp_config_path: str,
        model_platform: str = "openai",
        model_type: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        thought_logger: Optional[ThoughtProcessLogger] = None
    ):
        self.mcp_config_path = mcp_config_path
        self.model_platform = model_platform
        self.model_type = model_type
        self.temperature = temperature
        self.thought_logger = thought_logger
        self.mcp_toolkit = None
        self.agent = None
        
        if self.thought_logger:
            self.thought_logger.log_step("INITIALIZATION", "Initializing Weather Anomaly Agent Stageof Yogyakarta...")
    
    async def initialize(self):
        """Initialize MCP toolkit and CAMEL agent asynchronously"""
        try:
            # Initialize MCP toolkit
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", f"Loading MCP config: {self.mcp_config_path}")
            
            self.mcp_toolkit = MCPToolkit(config_path=self.mcp_config_path)
            
            # Connect to MCP server
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", "Connecting to MCP server...")
            
            await self.mcp_toolkit.connect()
            
            # Get tools
            tools = self.mcp_toolkit.get_tools()
            tool_names = [tool.get_function_name() for tool in tools]
            
            if self.thought_logger:
                self.thought_logger.log_step(
                    "TOOL_DISCOVERY",
                    f"Connected! Available tools: {', '.join(tool_names)}"
                )
            
            # Create CAMEL agent
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", "Creating CAMEL agent...")
            
            self.agent = self._create_agent(tools)
            
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", "Agent DIY Yogyakarta initialization complete")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during initialization: {e}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def _create_agent(self, tools):
        """Create the CAMEL agent with MCP tools."""
        try:

            # platform_map = {
            #     "gemini": ModelPlatformType.GEMINI,
            #     "openai": ModelPlatformType.OPENAI,
            #     "anthropic": ModelPlatformType.ANTHROPIC,
            #     "ollama": ModelPlatformType.OLLAMA,
            # }
            
            # platform_type = platform_map.get(self.model_platform.lower(), ModelPlatformType.OPENAI)
            
            # model = ModelFactory.create(
            #     model_platform=platform_type,
            #     model_type=self.model_type,
            #     model_config_dict={"temperature": self.temperature}
            # )
            
            system_message = """You are a weather sensor Anomaly Investigation Agent to publish findings of a potential sensor malfunction. Systematically analyze sensor anomalies and publish confirmed ones.

                PROCESS:
                1) IDENTIFY: Extract top 3 anomalous features from provided data

                2) RETRIEVE: Get latest 20 records for ALL parameters using get_data_from_db(features="tt,rh,pp,ws,wd,sr,rr", limit=20)

                3) IMMEDIATE SENSOR FAULT CHECK:
                - Invalid sentinels (e.g., -9999, 9999): PUBLISH immediately
                - Physically impossible values: PUBLISH immediately
                    * rh: must be 0-100%, values at exact 0% or 100% are highly suspicious
                    * wd: must be 0-360°
                    * tt: must be within reasonable range for location (-50°C to 60°C)
                    * ws: cannot be negative
                    * pp: typical range 950-1050 hPa
                - Stuck readings: same exact value for >6 intervals (60min): PUBLISH immediately

                4) CORRELATION CHECK (secondary validation):
                For features not caught above, check correlations:
                - tt (Temperature): Should correlate negatively with rh, positively with sr
                - rh (Humidity): Should correlate negatively with tt, positively with rr
                - ws (Wind Speed): Should correlate negatively with pp
                - pp (Pressure): Should correlate negatively with ws
                - sr (Solar Radiation): can change rapidly (300 units/10min is normal); low at night is expected
                - wd (Wind Direction): can change rapidly; do NOT use correlation; only check range (0-360°)
                
                Note: Broken correlations SUPPORT anomaly detection but intact correlations do NOT rule out sensor faults

                5) ANALYZE: Compare current vs historical (each record = 10 min interval):
                - Deviation magnitude and direction
                - Pattern: spike/drop/gradual/stuck over how many intervals
                - Correlation status with related parameters

                6) DECIDE: 
                Publish anomaly if you are CONFIDENT > 80% based on:
                - Sentinel values or physically impossible 
                - Stuck readings (>6 intervals) 
                - Large deviation (>3 std dev) 
                

                7) PUBLISH: Use send_anomalous_status with:
                - anomalous_features: {feature_code: value}
                - trend_analysis: {feature_code: "SHORT analysis including: type of fault, magnitude, timeframe, correlation status"}

                EXAMPLE:
                {
                "wd": "Invalid sentinel value -9999.0° - sensor communication failure",
                "rh": "Physically impossible 0% reading - sensor fault. Expected 48-56%, dropped 100% instantly. Correlation with tt appears intact but this doesn't rule out sensor malfunction.",
                "ws": "Stuck at 0.6 m/s for 60min (6 intervals), 88% below avg. Correlation with pp broken - sensor likely stuck/failed."
                }

                CRITICAL RULES:
                - ALWAYS publish confirmed sensor faults only - avoid false alarms
                - ALWAYS publish sentinel values (-9999, 9999, etc.) - these are definitive sensor faults
                - ALWAYS publish physically impossible values (rh=0%, rh=100%, wd outside 0-360°)
                - ALWAYS publish stuck readings (same value >6 intervals)
                - Intact correlations do NOT prove sensor is working - a broken sensor can show correlation by coincidence
                - For wind direction (wd): only check if value is valid (0-360°); ignore all other analyses
                - Only report "Possible false alarm" when deviation is moderate AND correlations are intact AND value is physically possible
                """

            agent = ChatAgent(
                system_message=system_message,
                model=ollama_model,
                tools=[*tools]  # Unpack the tools list
            )
            
            return agent
            
        except Exception as e:
            error_msg = f"Error creating agent: {e}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            import traceback
            traceback.print_exc()
            return None
    
    async def analyze(self, anomaly_data):
        """Analyze anomaly data and generate investigation report."""
        if not self.agent:
            return "Error: Agent not initialized properly. Call initialize() first."
        
        try:
            # Handle both dict and string inputs
            if isinstance(anomaly_data, dict):
                anomaly_json = json.dumps(anomaly_data, indent=2)
                data_dict = anomaly_data
            elif isinstance(anomaly_data, str):
                data_dict = json.loads(anomaly_data)
                anomaly_json = anomaly_data
            else:
                return "Error: Invalid input type. Expected dict or JSON string."
            
            # Log analysis start
            if self.thought_logger:
                self.thought_logger.log_analysis_start(data_dict)
                self.thought_logger.log_reasoning(
                    "Sending query to CAMEL agent with anomaly data for systematic analysis"
                )
            
            # Create user message
            user_message = f"""Please analyze the following weather anomaly data systematically:

                                {anomaly_json}

                                Follow your investigation process step by step."""
            
            # Get response from agent (async)
            if self.thought_logger:
                self.thought_logger.log_step("AGENT_PROCESSING", "Waiting for agent response...")
            
            response = await self.agent.astep(BaseMessage.make_user_message(
                role_name="User",
                content=user_message
            ))
            
            # Extract response content
            if hasattr(response, 'msgs') and response.msgs:
                result = response.msgs[0].content
            elif hasattr(response, 'msg'):
                result = response.msg.content
            else:
                result = str(response)
            
            # Log tool calls if available
            if hasattr(response, 'info') and 'tool_calls' in response.info:
                for tool_call in response.info['tool_calls']:
                    if self.thought_logger:
                        self.thought_logger.log_tool_call(tool_call.tool_name, tool_call.args)
                        self.thought_logger.log_tool_result(tool_call.tool_name, str(tool_call.result))
            
            # Log COMPLETE analysis result (not just summary)
            if self.thought_logger:
                self.thought_logger.log_analysis_end(result)
            
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {e}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"Error during analysis: {str(e)}\n\nDetails:\n{error_details}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            return error_msg
    
    async def cleanup(self):
        """Clean up resources"""
        if self.mcp_toolkit:
            await self.mcp_toolkit.disconnect()
            if self.thought_logger:
                self.thought_logger.log_step("CLEANUP", "Disconnected from MCP server")


    def analyze_sync(self, anomaly_data):
        """Synchronous wrapper for analyze - thread-safe with fresh connection"""
        async def analyze_with_connection():
            # Temporarily disconnect and reconnect MCP for this thread
            was_connected = self.mcp_toolkit is not None
            
            if was_connected:
                # Create fresh MCP connection for this thread
                temp_toolkit = MCPToolkit(config_path=self.mcp_config_path)
                await temp_toolkit.connect()
                
                # Temporarily swap toolkits
                old_toolkit = self.mcp_toolkit
                self.mcp_toolkit = temp_toolkit
                
                # Get fresh tools
                tools = temp_toolkit.get_tools()
                self.agent = self._create_agent(tools)
            
            try:
                # Run analysis
                result = await self.analyze(anomaly_data)
                return result
            finally:
                if was_connected:
                    # Cleanup temp connection
                    await temp_toolkit.disconnect()
                    # Restore original
                    self.mcp_toolkit = old_toolkit
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(analyze_with_connection())
        finally:
            loop.close()

async def run_agent(args):
    """Run the agent with given arguments"""
    try:
        # Create thought process logger (silent, file only)
        thought_logger = ThoughtProcessLogger(log_file=args.log_file)
        
        print(f"{CYAN}Initializing agent...{RESET}")
        
        agent = WeatherAnomalyAgent(
            mcp_config_path=args.mcp_config,
            model_platform=args.model_platform,
            model_type=args.model_type,
            temperature=args.temperature,
            thought_logger=thought_logger
        )
        
        # Initialize agent (async)
        if not await agent.initialize():
            print(f"{RED}Error: Failed to initialize agent{RESET}")
            return
        
        print(f"{GREEN}✓ Agent ready{RESET}")
        print(f"{CYAN}Logs: {args.log_file}{RESET}\n")
        
        try:
            # Handle file input
            if args.file:
                try:
                    with open(args.file, 'r') as f:
                        data = f.read().strip()
                    print(f"{YELLOW}Analyzing...{RESET}\n")
                    result = await agent.analyze(data)
                    print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                    print(result)
                    print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
                except FileNotFoundError:
                    print(f"{RED}Error: File {args.file} not found{RESET}")
                except Exception as e:
                    print(f"{RED}Error reading file: {e}{RESET}")
            
            # Handle JSON input
            elif args.json:
                print(f"{YELLOW}Analyzing...{RESET}\n")
                result = await agent.analyze(args.json)
                print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                print(result)
                print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
            
            # Interactive mode
            else:
                print("=" * 60)
                print("Weather Anomaly Investigation - Interactive Mode")
                print("=" * 60)
                print("Commands: 'quit', 'help', 'test', or paste JSON data")
                print("=" * 60)
                
                while True:
                    try:
                        user_input = input(f"\n{BOLD}{GREEN}> {RESET}").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print(f"\n{CYAN}Goodbye!{RESET}\n")
                            break
                        
                        if user_input.lower() == 'help':
                            sample = {
                                "timestamp": "2024-09-25T14:30:00",
                                "weather_data": {
                                    "tt": 35.2,
                                    "rh": 25.0,
                                    "pp": 1008.5,
                                    "ws": 15.8,
                                    "wd": 270,
                                    "sr": 1200.0,
                                    "rr": 0.0
                                },
                                "anomaly_scores": {
                                    "tt": 0.95,
                                    "rh": 0.85,
                                    "ws": 0.75
                                }
                            }
                            print(f"\n{CYAN}Sample JSON format:{RESET}")
                            print(json.dumps(sample, indent=2))
                            continue
                        
                        if user_input.lower() == 'test':
                            test_data = {
                                "timestamp": "2024-09-25T14:30:00",
                                "weather_data": {
                                    "tt": 38.5,
                                    "rh": 15.0,
                                    "pp": 1008.5,
                                    "ws": 8.2,
                                    "wd": 270,
                                    "sr": 1200.0,
                                    "rr": 0.0
                                },
                                "anomaly_scores": {
                                    "tt": 0.95,
                                    "rh": 0.92,
                                    "ws": 0.45
                                }
                            }
                            print(f"\n{YELLOW}Analyzing...{RESET}\n")
                            result = await agent.analyze(test_data)
                            print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                            print(result)
                            print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
                            continue
                        
                        if user_input:
                            print(f"\n{YELLOW}Analyzing...{RESET}\n")
                            result = await agent.analyze(user_input)
                            print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                            print(result)
                            print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
                        else:
                            print(f"{YELLOW}Please provide JSON data or type 'help'{RESET}")
                            
                    except KeyboardInterrupt:
                        print(f"\n\n{CYAN}Goodbye!{RESET}\n")
                        break
                    except Exception as e:
                        print(f"{RED}Error: {e}{RESET}")
            
            # Save summary if requested
            if args.save_summary:
                thought_logger.save_summary(args.save_summary)
                print(f"{GREEN}Summary saved to: {args.save_summary}{RESET}")
        
        finally:
            # Always cleanup
            await agent.cleanup()
                    
    except Exception as e:
        print(f"{RED}Fatal error: {e}{RESET}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with interactive and non-interactive modes."""
    parser = argparse.ArgumentParser(
        description="Weather Anomaly Investigation Agent with CAMEL AI + MCP Tools"
    )
    parser.add_argument(
        "--json", "-j", 
        help="JSON anomaly data to analyze"
    )
    parser.add_argument(
        "--file", "-f", 
        help="JSON file containing anomaly data"
    )
    parser.add_argument(
        "--mcp_config", "-c",
        default="config/weather_anomaly.json",
        help="Path to MCP config file (default: config/weather_anomaly.json)"
    )
    parser.add_argument(
        "--model_platform", "-p",
        default="openai",
        choices=["gemini", "openai", "anthropic", "ollama"],
        help="Model platform (default: openai)"
    )
    parser.add_argument(
        "--model_type", "-m",
        default="gpt-4o-mini",
        help="Model type (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--log_file", "-l",
        default="logs/thought_process.log",
        help="Path to log file for thought process (default: logs/thought_process.log)"
    )
    parser.add_argument(
        "--save_summary", "-ss",
        help="Save thought process summary to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run async function
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()