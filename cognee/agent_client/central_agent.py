"""
central_agent.py

Main agent orchestration with direct Cognee integration.
Multi-agent system with Weather Agent and Sensor Agent.

DIRECT COGNEE VERSION - No MCP needed
"""

from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage
from camel.tasks import Task
import asyncio
from dotenv import load_dotenv
import os
import sys
import warnings
import logging

# Import your existing modules
from tools import *
from prompt_template import *

# Import direct Cognee tools
from cognee_direct_tools import get_cognee_tools

# Suppress async warnings for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv()

# Initialize local LLM model
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11112/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16384,
    },
)


def setup_agents():
    """
    Setup all agents including custom Workforce coordinator and task agents.
    
    Returns:
        tuple: (workforce, weather_agent, sensor_agent)
    """
    
    print("\n" + "="*70)
    print("INITIALIZING MULTI-AGENT SYSTEM")
    print("="*70)

    ##======================== Weather Agent ========================
    print("\n[1/4] Setting up Weather Agent...")
    weather_tool = FunctionTool(get_weather_param)
    
    weather_agent = ChatAgent(
        system_message=WEATHER_AGENT_PROMPT,
        tools=[weather_tool],
        model=ollama_model,
    )
    print("✓ Weather Agent ready")
    
    ##======================== Sensor Agent (Direct Cognee) ========================
    print("\n[2/4] Setting up Sensor Agent with Direct Cognee Tools...")
    
    # Get Cognee tools directly (no MCP)
    sensor_tools = get_cognee_tools()
    
    # Create Sensor agent with Cognee tools
    sensor_agent = ChatAgent(
        system_message=SENSOR_AGENT_PROMPT,
        tools=sensor_tools,
        model=ollama_model,
    )
    print("✓ Sensor Agent ready with Direct Cognee tools")
    
    ##======================== Task Agent (Simplified) ========================
    print("\n[3/4] Setting up Task Agent (simplified task decomposition)...")
    
    task_agent_prompt = """You are a Task Decomposition Agent. Your job is to break down user queries into simple, natural subtasks.

**CRITICAL RULES:**
1. Keep subtasks SHORT and CONVERSATIONAL (1-2 sentences max)
2. NEVER add JSON schemas, data structures, or formatting requirements
3. NEVER over-specify HOW to format responses
4. Focus only on WHAT information is needed
5. Let worker agents decide their own output format
6. Use natural language, not technical specifications

**Available Workers:**
- Weather Analyst: Retrieves weather data (temperature, precipitation, wind, etc.)
- Sensor Monitor: Searches sensor specifications and knowledge base

**Examples of GOOD subtasks:**
❌ BAD: "Using available meteorological data sources, retrieve 15-minute precipitation accumulations for latitude -16.52... [detailed JSON schema]"
✅ GOOD: "Get precipitation data for latitude -16.52, longitude 13.41 on January 15, 2025 between 3:20-4:00 AM UTC"

❌ BAD: "Query the knowledge graph with the following parameters: {location: {...}, temporal_window: {...}}..."
✅ GOOD: "Search for sensor anomaly reports near latitude -16.52, longitude 13.41 on January 15, 2025"

**Your output should be:**
- A simple list of subtasks
- Each subtask is a natural language instruction
- No formatting requirements
- No technical jargon unless necessary

When a query only needs one agent, create just ONE subtask. Don't overcomplicate!"""

    task_agent = ChatAgent(
        system_message=task_agent_prompt,
        model=ollama_model,
    )
    print("✓ Task Agent ready (simplified mode)")
    
    ##======================== Coordinator Agent (Simplified) ========================
    print("\n[4/4] Setting up Coordinator Agent...")
    
    coordinator_agent_prompt = """You are a Workforce Coordinator. Your job is to:
1. Assign subtasks to the appropriate worker agents
2. Collect their responses
3. Synthesize a clear, natural language final answer

**CRITICAL RULES:**
1. Provide responses in NATURAL LANGUAGE, not JSON
2. Synthesize information from multiple workers into a coherent answer
3. Be conversational and helpful
4. Only include technical details if specifically requested
5. Focus on answering the user's original question directly

**Available Workers:**
- Weather Analyst: Weather data queries
- Sensor Monitor: Sensor and knowledge base queries

**Response Format:**
- Start with a direct answer to the user's question
- Include relevant details from worker responses
- Keep it concise but complete
- Use natural paragraphs, not bullet points unless appropriate

**Example:**
User: "What was the precipitation on January 15?"
Your response: "Based on weather data, there was no precipitation recorded at the specified location on January 15, 2025 between 3:20-4:00 AM UTC. The weather conditions were dry during this 40-minute period, with 0.0mm of rainfall detected by the Open-Meteo monitoring system."

NOT this: "Here is the JSON output: {precipitation_total_mm: 0.0, ...}"

Keep responses human-friendly and conversational!"""

    coordinator_agent = ChatAgent(
        system_message=coordinator_agent_prompt,
        model=ollama_model,
    )
    print("✓ Coordinator Agent ready")
    
    ##======================== Setup Workforce ========================
    print("\n" + "="*70)
    print("Building Workforce with custom agents...")
    print("="*70)
    
    workforce = Workforce(
        description='Multi-Agent Weather & Knowledge System',
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
        description='Searches sensor specifications, technical documentation, and knowledge base for sensor-related information and anomaly reports'
    )
    
    print("\n✓ Workforce ready with custom coordination:")
    print("   - Task Agent: Simplified task decomposition (no JSON schemas)")
    print("   - Coordinator Agent: Natural language synthesis")
    print("   - Weather Analyst: Weather data analysis")
    print("   - Sensor Monitor: Direct Cognee knowledge graph")
    print("="*70)
    print()
    
    return workforce, weather_agent, sensor_agent


def test_individual_agents(weather_agent, sensor_agent):
    """Test individual agents before workforce integration."""
    
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL AGENTS")
    print("="*70)
    
    # Test Weather Agent
    print("\n--- Testing Weather Agent ---")
    weather_msg = BaseMessage.make_user_message(
        role_name="User",
        content="What was the temperature in latitude -16.52, longitude 13.41 on 2025-01-15 at 3:20 to 04:00 AM UTC?"
    )
    print("Calling weather agent...")
    weather_response = weather_agent.step(weather_msg)
    print(f"\nWeather Agent Response:")
    print(f"{weather_response.msgs[0].content}")
    
    # Test Sensor Agent
    print("\n" + "-"*70)
    print("--- Testing Sensor Agent ---")
    sensor_msg = BaseMessage.make_user_message(
        role_name="User",
        content="what is the operational range of HMP155"
    )
    print("Calling sensor agent...")
    sensor_response = sensor_agent.step(sensor_msg)
    print(f"\nSensor Agent Response:")
    print(f"{sensor_response.msg.content}")
    
    print("\n" + "="*70)


def test_workforce(workforce):
    """Test workforce with a complex task."""
    
    print("\n" + "="*70)
    print("TESTING WORKFORCE COLLABORATION")
    print("="*70)

    task = "What is the temperature at latitude -16.52, longitude 13.41 on January 15, 2025 at 3:20 to 04:00 AM UTC? Is this range suitable for operating the HMP155 sensor?"
    
    print(f"\nTask: {task}")
    print("\nProcessing...\n")
    
    try:
        result = workforce.process_task(Task(content=task))
        
        print(f"\n✓ Workforce Result:")
        print("="*70)
        
        if result.result:
            print(result.result)
        else:
            print("No result generated")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Workforce execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)


def interactive_mode(weather_agent, sensor_agent, workforce):
    """Interactive mode for testing agents."""
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nCommands:")
    print("  1 - Query Weather Agent")
    print("  2 - Query Sensor Agent")
    print("  3 - Query Workforce")
    print("  q - Quit")
    print("="*70)
    
    while True:
        try:
            choice = input("\nSelect option: ").strip()
            
            if choice == 'q':
                break
            
            if choice in ['1', '2', '3']:
                query = input("Enter your query: ").strip()
                
                if not query:
                    continue
                
                msg = BaseMessage.make_user_message(
                    role_name="User",
                    content=query
                )
                
                if choice == '1':
                    print("\n[Weather Agent]")
                    response = weather_agent.step(msg)
                    print(f"\n{response.msgs[0].content}")
                
                elif choice == '2':
                    print("\n[Sensor Agent]")
                    response = sensor_agent.step(msg)
                    print(f"\n{response.msg.content}")
                
                elif choice == '3':
                    print("\n[Workforce]")
                    result = workforce.process_task(Task(content=query))
                    print(f"\n{result.result if result.result else 'No result'}")
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution flow."""
    
    try:
        print("\n" + "="*70)
        print("STARTING MULTI-AGENT SYSTEM")
        print("="*70)
        
        # Setup all agents
        workforce, weather_agent, sensor_agent = setup_agents()
        
        # Run individual agent tests
        #print("\n[Phase 1] Testing individual agents...")
        #test_individual_agents(weather_agent, sensor_agent)
        
        # Test workforce collaboration
        print("\n[Phase 2] Testing workforce collaboration...")
        test_workforce(workforce)
        
        # Interactive mode (optional - uncomment to use)
        # print("\n[Phase 3] Starting interactive mode...")
        # interactive_mode(weather_agent, sensor_agent, workforce)
        
        print("\n" + "="*70)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
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