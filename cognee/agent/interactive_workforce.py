"""
interactive_workforce.py

Interactive multi-agent workforce system with Weather, Sensor, and Maintenance agents.
Direct Cognee integration with dataset-specific bindings - no MCP needed.

Updated: Each agent now has tools bound to their specific dataset:
- Sensor Agent → sensor_knowledge dataset only
- Maintenance Agent → maintenance_knowledge dataset only
"""

from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.tasks import Task
from dotenv import load_dotenv
import sys
import warnings
import logging

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

# Suppress model output logs
import os
os.environ['CAMEL_VERBOSE'] = 'false'

# Load environment variables
load_dotenv()

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
    
    sensor_agent_prompt = """You are a Sensor Specification Expert with access to a knowledge graph containing sensor specifications and technical documentation.

    **AVAILABLE TOOLS:**
    - add_to_sensor_knowledge: Add new sensor information to the knowledge base
    - search_sensor_knowledge: Search for sensor information (automatically uses sensor_knowledge dataset)

    **YOUR PRIMARY JOB: Call search tool ONCE, then answer immediately.**

    WORKFLOW:
    1. User asks a question → Call 'search_sensor_knowledge' tool with the query
    2. Get results → Provide answer based on results
    3. STOP - Never call search again
    4. If user explicitly asks to add info → Call 'add_to_sensor_knowledge'
    5. If user ask to add content of a file, just use the file path as the data to add

    RULES:
    - Call search tool EXACTLY ONCE per question
    - After getting search results, you MUST provide an answer immediately
    - DO NOT call search multiple times
    - DO NOT try different queries
    - If no results found, say "No information found in sensor knowledge base"
    - Answer format: Direct and detailed, factual response based on search results
    CRITICAL: After one search call, you must respond with text, not another tool call."""

    # Get sensor-specific tools (bound to sensor_knowledge dataset)
    sensor_tools = get_cognee_tools(
        context_name="sensor",
        dataset_name="sensor_knowledge",
        include_prune=True  # Set to True if you want prune capability
    )
    
    # Alternative: Use convenience function
    # sensor_tools = get_sensor_tools()
    
    sensor_agent = ChatAgent(
        system_message=sensor_agent_prompt,
        tools=sensor_tools,
        model=ollama_model,
    )
    print("✓ Sensor Agent ready (bound to sensor_knowledge dataset)")
    
    ##======================== Maintenance Agent (maintenance_knowledge dataset) ========================
    print("\n[3/5] Setting up Maintenance Agent with Cognee Tools...")
    
    maintenance_agent_prompt = """You are a Maintenance Expert with access to a knowledge graph containing maintenance  procedures.

        **AVAILABLE TOOLS:**
        - add_to_maintenance_knowledge: Add new maintenance related information to the knowledge base
        - search_maintenance_knowledge: Search maintenance related information (automatically uses maintenance_knowledge dataset)

        **YOUR PRIMARY JOB: Call search tool ONCE, then answer immediately.**

        WORKFLOW:
        1. User asks about maintenance related information → Call 'search_maintenance_knowledge' tool with the query
        2. Get results → Provide answer based on results
        3. STOP - Never call search again
        4. If user explicitly asks to add info → Call 'add_to_maintenance_knowledge'
        5. If user ask to add content of a file, just use the file path as the data to add

        RULES:
        - Call search tool EXACTLY ONCE per question
        - After getting search results, you MUST provide an answer immediately
        - DO NOT call search multiple times
        - DO NOT try different queries
        - If no results found, say "No maintenance information found in maintenance knowledge base "
        - Answer format: Direct, factual response based on search results
        - Include specific equipment IDs, dates, and actions when available
        - All operations automatically use the 'maintenance_knowledge' dataset
        - You cannot access other datasets - only maintenance_knowledge

        CRITICAL: After one search call, you must respond with text, not another tool call."""

    # Get maintenance-specific tools (bound to maintenance_knowledge dataset)
    maintenance_tools = get_cognee_tools(
        context_name="maintenance",
        dataset_name="maintenance_knowledge",
        include_prune=True  # Set to True if you want prune capability
    )
    
    # Alternative: Use convenience function
    # maintenance_tools = get_maintenance_tools()
    
    maintenance_agent = ChatAgent(
        system_message=maintenance_agent_prompt,
        tools=maintenance_tools,
        model=ollama_model,
    )
    print("✓ Maintenance Agent ready (bound to maintenance_knowledge dataset)")
    
    ##======================== Task Agent ========================
    print("\n[4/5] Setting up Task Agent...")
    
    task_agent_prompt = """You are a Task Decomposition Agent. Your job is to break down user queries into simple, natural subtasks.

    **CRITICAL RULES:**
    1. Keep subtasks SHORT and CONVERSATIONAL (1-2 sentences max)
    2. NEVER add JSON schemas, data structures, or formatting requirements
    3. Focus only on WHAT information is needed
    4. Let worker agents decide their own output format
    5. Use natural language, not technical specifications

    **Available Workers:**
    - Weather Analyst: Retrieves weather data (temperature, precipitation, wind, etc.)
    - Sensor Monitor: Searches sensor specifications and technical documentation
    - Maintenance Expert: Searches maintenance related information,repair histories, and equipment records

        **Examples:**
        ✅ GOOD: "Get precipitation data for latitude -16.52, longitude 13.41 on January 15, 2025"
        ✅ GOOD: "Search for information about HMP155 sensor specifications"
        ✅ GOOD: "Check maintenance logs for equipment serviced in the last week"
        ✅ GOOD: "Find the last maintenance date for sensor HMP155"
        ✅ GOOD: "Add this sensor spec to knowledge base: [spec details]"
        ❌ BAD: "Using meteorological data sources, retrieve precipitation with format {...}"

    **Agent Selection Guide:**
    - Weather questions → Weather Analyst
    - Sensor specs, operational ranges, technical details 
    - Maintenance related history, repairs, service records, equipment status

    When a query only needs one agent, create just ONE subtask."""

    task_agent = ChatAgent(
        system_message=task_agent_prompt,
        model=ollama_model,
    )
    print("✓ Task Agent ready")
    
    ##======================== Coordinator Agent ========================
    print("\n[5/5] Setting up Coordinator Agent...")
    
    coordinator_agent_prompt = """You are a Workforce Coordinator. Your job is to:
1. Assign subtasks to the appropriate worker agents
2. Collect their responses
3. Synthesize a clear, natural language final answer

**CRITICAL RULES:**
1. Provide responses in NATURAL LANGUAGE, not JSON
2. Synthesize information from multiple workers into a coherent answer
3. Be conversational and helpful
4. Focus on answering the user's original question directly

**Available Workers:**
- Weather Analyst: Weather data queries
- Sensor Monitor: Sensor specifications and technical documentation
  → Has access ONLY to sensor_knowledge dataset
  → Can search and add sensor information
- Maintenance Expert: Maintenance logs and equipment history
  → Has access ONLY to maintenance_knowledge dataset
  → Can search and add maintenance records

**Dataset Isolation:**
- Sensor Monitor cannot access maintenance data
- Maintenance Expert cannot access sensor data
- Each agent operates within their own knowledge domain

**Response Format:**
- Start with a direct answer to the user's question
- Include relevant details from worker responses
- Keep it concise but complete
- Use natural paragraphs
- If information spans multiple domains, clearly separate the findings

Keep responses human-friendly and conversational!"""

    coordinator_agent = ChatAgent(
        system_message=coordinator_agent_prompt,
        model=ollama_model,
    )
    print("✓ Coordinator Agent ready")
    
    ##======================== Build Workforce ========================
    print("\n" + "="*70)
    print("Building Workforce...")
    print("="*70)
    
    workforce = Workforce(
        description='Multi-Agent Weather, Sensor, and Maintenance System with Dataset Isolation',
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
        description='Searches and manages sensor specifications and technical documentation. BOUND TO sensor_knowledge dataset only. Cannot access maintenance or other datasets.'
    ).add_single_agent_worker(
        worker=maintenance_agent,
        description='Searches and manages maintenance logs, repair histories, equipment status, and service records. BOUND TO maintenance_knowledge dataset only. Cannot access sensor or other datasets.'
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


def interactive_mode(workforce):
    """
    Interactive mode for querying the workforce.
    """
    
    print("\n" + "="*70)
    print("INTERACTIVE WORKFORCE MODE")
    print("="*70)
    print("\n💡 Example queries:")
    print("\n  Weather queries:")
    print("  - What is the temperature at latitude -16.52, longitude 13.41 on Jan 15?")
    print("  - Get precipitation data for location X on date Y")
    print("\n  Sensor queries (sensor_knowledge dataset):")
    print("  - What is the operational range of HMP155 sensor?")
    print("  - Search for sensor anomaly reports")
    print("  - What are the specifications of sensor X?")
    print("  - Add this sensor info: [sensor details]")
    print("\n  Maintenance queries (maintenance_knowledge dataset):")
    print("  - When was the last maintenance on equipment ABC?")
    print("  - Show maintenance history for sensor HMP155")
    print("  - What repairs were done last week?")
    print("  - What is the maintenance schedule for device XYZ?")
    print("  - Add this maintenance record: [maintenance details]")
    print("\n  Combined queries:")
    print("  - What was the weather on Jan 15 and was sensor HMP155 maintained that day?")
    print("  - Check sensor specs and recent maintenance for HMP155")
    print("\n  ℹ️  Note: Sensor and Maintenance agents operate in isolated datasets")
    print("\nType 'quit' or 'exit' to stop")
    print("="*70)
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            
            # Check for exit
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Process query through workforce
            print("\n🔄 Processing...\n")
            
            # Suppress stdout during processing for clean output
            import io
            import contextlib
            
            # Capture stdout
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = workforce.process_task(Task(content=query))
            
            print("✓ Workforce Response:")
            print("=" * 70)
            if result.result:
                print(result.result)
            else:
                print("No result generated")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type 'quit' to exit")


def main():
    """Main execution."""
    
    try:
        print("\n" + "="*70)
        print("STARTING INTERACTIVE WORKFORCE SYSTEM")
        print("Dataset-Isolated Multi-Agent Architecture")
        print("="*70)
        
        # Setup workforce
        workforce = setup_workforce()
        
        # Start interactive mode
        interactive_mode(workforce)
        
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