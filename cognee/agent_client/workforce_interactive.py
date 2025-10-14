"""
interactive_workforce.py

Interactive multi-agent workforce system with Weather and Sensor agents.
Direct Cognee integration - no MCP needed.
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

# Import direct Cognee tools
from cognee_direct_tools import get_cognee_tools

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
    url="http://10.33.205.34:11112/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16384,
    },
)


def setup_workforce():
    """
    Setup workforce with Weather and Sensor agents.
    
    Returns:
        Workforce: Configured workforce ready for use
    """
    
    print("\n" + "="*70)
    print("INITIALIZING MULTI-AGENT WORKFORCE")
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
    
    ##======================== Sensor Agent ========================
    print("\n[2/4] Setting up Sensor Agent with Cognee Tools...")
    
    # Update sensor agent prompt to prevent looping
    sensor_agent_prompt = """You are a Sensor Specification Expert with access to a knowledge graph.

**CRITICAL: STOP AFTER ONE SEARCH**

Your workflow when user ask a question:
1. User asks a question
2. Call the 'search' tool ONCE
3. Read the results
4. Provide your answer IMMEDIATELY
5. STOP - Do not search again

Your workflow when user want you to add to your knowledge base:
1. User asks you to add information
2. Call the 'cognify' tool ONCE
3. STOP - Do not cognify again

Rules:
- Maximum ONE search call per question
- If results are found: Answer based on those results
- If no results: Say "I don't have information about that"
- NEVER call search multiple times
- NEVER try different search queries
- After getting results, you MUST answer, not search again


Do NOT keep searching. One search, one answer.
Just return what you found from search tool, do not retry to call it and do not debate the answer."""

    sensor_tools = get_cognee_tools()
    
    sensor_agent = ChatAgent(
        system_message=sensor_agent_prompt,
        tools=sensor_tools,
        model=ollama_model,
    )
    print("✓ Sensor Agent ready")
    
    ##======================== Task Agent ========================
    print("\n[3/4] Setting up Task Agent...")
    
    task_agent_prompt = """You are a Task Decomposition Agent. Your job is to break down user queries into simple, natural subtasks.

**CRITICAL RULES:**
1. Keep subtasks SHORT and CONVERSATIONAL (1-2 sentences max)
2. NEVER add JSON schemas, data structures, or formatting requirements
3. Focus only on WHAT information is needed
4. Let worker agents decide their own output format
5. Use natural language, not technical specifications

**Available Workers:**
- Weather Analyst: Retrieves weather data (temperature, precipitation, wind, etc.)
- Sensor Monitor: Searches sensor specifications and knowledge base

**Examples:**
✅ GOOD: "Get precipitation data for latitude -16.52, longitude 13.41 on January 15, 2025"
✅ GOOD: "Search for information about HMP155 sensor specifications"
❌ BAD: "Using meteorological data sources, retrieve precipitation with format {...}"

When a query only needs one agent, create just ONE subtask."""

    task_agent = ChatAgent(
        system_message=task_agent_prompt,
        model=ollama_model,
    )
    print("✓ Task Agent ready")
    
    ##======================== Coordinator Agent ========================
    print("\n[4/4] Setting up Coordinator Agent...")
    
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
- Sensor Monitor: Sensor and knowledge base queries

**Response Format:**
- Start with a direct answer to the user's question
- Include relevant details from worker responses
- Keep it concise but complete
- Use natural paragraphs

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
    
    print("\n✓ Workforce ready:")
    print("   - Weather Analyst: Weather data analysis")
    print("   - Sensor Monitor: Cognee knowledge graph")
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
    print("  - What is the operational range of HMP155 sensor?")
    print("  - Get weather data for latitude -16.52, longitude 13.41 on Jan 15, 2025")
    print("  - Search for sensor anomaly reports")
    print("  - What was the temperature and precipitation on January 15?")
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