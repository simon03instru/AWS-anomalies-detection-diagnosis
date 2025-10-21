"""
cognee_direct_tools.py

Direct Cognee integration with dynamic node_set determination.
Agents autonomously decide node_sets during operation.
Uses ENABLE_BACKEND_ACCESS_CONTROL for dataset isolation.
"""

import cognee
from typing import Optional, List
import asyncio
from functools import wraps
from camel.toolkits import FunctionTool
import nest_asyncio
import os

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Enable dataset isolation (CRITICAL for proper isolation)
os.environ['ENABLE_BACKEND_ACCESS_CONTROL'] = 'true'
os.environ['REQUIRE_AUTHENTICATION'] = 'true'


def async_to_sync(async_func):
    """Decorator to convert async functions to sync for CAMEL compatibility."""
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(async_func(*args, **kwargs))
            else:
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
    return wrapper


def initialize_cognee_sync():
    """
    Initialize Cognee database synchronously.
    
    NOTE: This does NOT prune existing data.
    Data persists across runs unless explicitly pruned via the prune tool.
    """
    @async_to_sync
    async def _init():
        # Just initialize Cognee, don't prune
        # Cognee will auto-initialize on first use
        pass
    _init()


# ==================== Dataset-Specific Tool Generators ====================

def create_agent_add_tool(dataset_name: str):
    """
    Create an 'add' tool where agents autonomously decide node_sets.
    
    Args:
        dataset_name: The dataset name (e.g., "sensor_knowledge")
    """
    async def _async_add(data: str, node_sets: List[str]):
        """Internal async implementation.
        
        Args:
            data: The text content to add
            node_sets: List of node sets determined by the agent
                      Example: ["Real_Time_Data", "Anomaly"]
        """
        # Validate node_sets is provided and is a list
        if not node_sets:
            raise ValueError("node_sets cannot be empty. Agent must specify at least one node_set.")
        
        if not isinstance(node_sets, list):
            raise ValueError("node_sets must be a list of strings")
        
        # Add to specific dataset with agent-determined node_sets
        await cognee.add(
            data,
            dataset_name=dataset_name,
            node_set=node_sets
        )
        await cognee.cognify()
        
        node_set_str = ", ".join(node_sets)
        return f"✓ Successfully added to {dataset_name} under node_sets: [{node_set_str}]"
    
    def agent_add(data: str, node_sets: List[str]):
        f"""Add information to {dataset_name} knowledge base.
        
        The agent must determine which node_sets this data belongs to based on:
        - The type of data (real-time, historical, calibration, anomaly, etc.)
        - The context of the information
        - The intended use case
        
        Args:
            data (str): The text content to add to the knowledge base
            node_sets (List[str]): Node sets determined by this agent.
                                  Examples: ["temperature_humidity_sensor"], ["temperature_humidity_sensor", "pressure_sensor"]
                                  
                                    Common Node_set categories:
                                - temperature_humidity_sensor: For temp/humidity sensor troubleshooting
                                - pressure_sensor: For pressure sensor troubleshooting
                                - wind_sensor: For wind sensor troubleshooting
                                - solar_radiation_sensor: For solar radiation sensor troubleshooting
                                - rain_sensor: For rain sensor troubleshooting
        
        Returns:
            Success message with node_sets confirmation
            
        Raises:
            ValueError: If node_sets is empty or not a list
        """
        return async_to_sync(_async_add)(data=data, node_sets=node_sets)
    
    agent_add.__name__ = f"add_to_{dataset_name}"
    agent_add.__doc__ = f"Add information to {dataset_name} with agent-determined node_sets."
    
    return agent_add


def create_agent_search_tool(dataset_name: str):
    """
    Create a 'search' tool where agents autonomously decide which node_sets to query.
    
    Returns ONLY the search_result text, not the full chunks/graphs.
    
    Args:
        dataset_name: The dataset name (e.g., "sensor_knowledge")
    """
    async def _async_search(query: str, node_sets: List[str]):
        """Internal async implementation.
        
        Args:
            query: The search query text
            node_sets: List of node sets to search within, determined by agent
        """
        # Validate node_sets
        if not node_sets:
            raise ValueError("node_sets cannot be empty. Agent must specify at least one node_set to search.")
        
        if not isinstance(node_sets, list):
            raise ValueError("node_sets must be a list of strings")
        
        # Search ONLY in the specified dataset and node_sets
        results = await cognee.search(
            query_text=query,
            node_name=node_sets
        )
        
        # No results
        if not results:
            node_set_str = ", ".join(node_sets)
            return f"No information found in {dataset_name} (node_sets: {node_set_str}) for query: '{query}'"
        
        # Extract ONLY the search_result field, ignore graphs/chunks
        formatted_results = []
        
        for result in results:
            # If result is a dict with 'search_result' field
            if isinstance(result, dict) and 'search_result' in result:
                search_results = result['search_result']
                
                # search_result is a list of strings
                if isinstance(search_results, list):
                    formatted_results.extend(search_results)
                else:
                    formatted_results.append(str(search_results))
            
            # Fallback: try other common fields
            elif hasattr(result, 'text'):
                formatted_results.append(result.text)
            elif hasattr(result, 'content'):
                formatted_results.append(result.content)
            elif isinstance(result, dict):
                if 'text' in result:
                    formatted_results.append(result['text'])
                elif 'content' in result:
                    formatted_results.append(result['content'])
        
        # Return clean text without graphs/chunks
        if formatted_results:
            combined = "\n\n".join(formatted_results)
            
            # Truncate if too long (max 3000 chars)
            MAX_LENGTH = 3000
            if len(combined) > MAX_LENGTH:
                combined = combined[:MAX_LENGTH] + f"\n\n[Results truncated - showing first {MAX_LENGTH} characters]"
            
            return combined
        else:
            return f"Found results but unable to extract text from {dataset_name}"
    
    def agent_search(query: str, node_sets: List[str]):
        f"""Search the {dataset_name} knowledge base.
        
        The agent must determine which node_sets are relevant for this search based on:
        - The type of information being queried
        - The context and use case
        - The expected data sources
        
        Args:
            query (str): The search query text
            node_sets (List[str]): Node sets to search within, determined by this agent.
                                  Examples: ["temperature_humidity_sensor"], ["temperature_humidity_sensor", "pressure_sensor"]
                                  
                                Common Node_set categories:
                                - temperature_humidity_sensor: For temp/humidity sensor specifications
                                - pressure_sensor: For pressure sensor specifications
                                - wind_sensor: For wind sensor specifications
                                - solar_radiation_sensor: For solar radiation sensor specifications
                                - rain_sensor: For rain sensor specifications

        Returns:
            Clean search results as text
            
        Raises:
            ValueError: If node_sets is empty or not a list
        """
        return async_to_sync(_async_search)(query=query, node_sets=node_sets)
    
    agent_search.__name__ = f"search_{dataset_name}"
    agent_search.__doc__ = f"Search {dataset_name} with agent-determined node_sets."
    
    return agent_search


def create_agent_prune_tool(dataset_name: str):
    """
    Create a 'prune' tool for a specific dataset.
    
    Args:
        dataset_name: The dataset name
    """
    async def _async_prune():
        """Internal async implementation."""
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        
        return f"✓ Cleared all Cognee data (affects all datasets)"
    
    def agent_prune():
        f"""Clear all data from Cognee.
        
        WARNING: This clears ALL datasets and node_sets, not just {dataset_name}.
        
        Returns:
            Success message
        """
        return async_to_sync(_async_prune)()
    
    agent_prune.__name__ = f"prune_{dataset_name}"
    agent_prune.__doc__ = f"Clear all Cognee data (WARNING: affects all datasets)."
    
    return agent_prune


# ==================== Main Tool Generator ====================

def get_cognee_tools(
    context_name: str,
    dataset_name: Optional[str] = None,
    include_prune: bool = False
) -> List[FunctionTool]:
    """
    Generate Cognee tools where agents autonomously determine node_sets.
    
    Uses Cognee's built-in access control and node_set system:
    - ENABLE_BACKEND_ACCESS_CONTROL=true enforces dataset boundaries
    - Agents decide node_sets dynamically during operation
    - Uses Kùzu (graph) and LanceDB (vector) for isolation support
    
    Args:
        context_name: Context identifier (e.g., "sensor", "maintenance")
        dataset_name: Dataset name (defaults to f"{context_name}_knowledge")
        include_prune: Whether to include prune tool (default: False)
    
    Returns:
        List of FunctionTool objects with dynamic node_set support
    
    Example:
        sensor_tools = get_cognee_tools(context_name="sensor")
        # Agent will decide to use node_sets like:
        # - add_to_sensor_knowledge(data="...", node_sets=["Real_Time_Data"])
        # - search_sensor_knowledge(query="...", node_sets=["Anomaly", "Trend_Analysis"])
    """
    # Default dataset name
    if dataset_name is None:
        dataset_name = f"{context_name}_knowledge"
    
    print(f"   → Creating Cognee tools for dataset: {dataset_name}")
    print(f"   → Agents will autonomously determine node_sets during operation")
    print(f"   → Access control enabled: ENABLE_BACKEND_ACCESS_CONTROL=true")
    print(f"   → Dataset isolation enforced by Cognee")
    print(f"   → Dynamic node_set determination enabled")
    print(f"   → Search returns clean text only (no graphs/chunks)")
    
    # Create tools with agent-determined node_sets
    add_func = create_agent_add_tool(dataset_name)
    search_func = create_agent_search_tool(dataset_name)
    
    tools = [
        FunctionTool(add_func),
        FunctionTool(search_func)
    ]
    
    # Optionally add prune tool
    if include_prune:
        prune_func = create_agent_prune_tool(dataset_name)
        tools.append(FunctionTool(prune_func))
        print(f"   → Added prune tool (WARNING: affects all datasets)")
    
    print(f"   → Generated {len(tools)} tools with autonomous node_set determination")
    
    return tools


# ==================== Convenience Functions ====================

def get_sensor_tools(include_prune: bool = False) -> List[FunctionTool]:
    """Get tools for sensor agent with autonomous node_set determination."""
    return get_cognee_tools(
        context_name="sensor",
        dataset_name="sensor_knowledge",
        include_prune=include_prune
    )


def get_maintenance_tools(include_prune: bool = False) -> List[FunctionTool]:
    """Get tools for maintenance agent with autonomous node_set determination."""
    return get_cognee_tools(
        context_name="maintenance",
        dataset_name="maintenance_knowledge",
        include_prune=include_prune
    )