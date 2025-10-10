"""
cognee_direct_tools.py

Direct Cognee integration with proper dataset isolation using built-in access control.
Uses ENABLE_BACKEND_ACCESS_CONTROL for true dataset isolation.
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
    Create an 'add' tool bound to a specific dataset.
    Uses Cognee's built-in dataset isolation.
    
    Args:
        dataset_name: The dataset name (e.g., "sensor_knowledge")
    """
    async def _async_add(data: str):
        """Internal async implementation."""
        # Add to specific dataset - Cognee will enforce isolation
        await cognee.add(
            data,
            dataset_name=dataset_name
        )
        await cognee.cognify()
        
        return f"✓ Successfully added to {dataset_name} (isolated dataset)"
    
    def agent_add(data: str):
        f"""Add information to {dataset_name} knowledge base.
        
        This data is stored in an isolated dataset with access control.
        Only searches specifying this dataset will find this data.
        
        Args:
            data: The text content to add to the knowledge base
        
        Returns:
            Success message confirming data was added
        """
        return async_to_sync(_async_add)(data=data)
    
    agent_add.__name__ = f"add_to_{dataset_name}"
    agent_add.__doc__ = f"Add information to {dataset_name} (isolated dataset)."
    
    return agent_add


def create_agent_search_tool(dataset_name: str):
    """
    Create a 'search' tool bound to a specific dataset.
    Uses Cognee's built-in dataset isolation.
    
    Args:
        dataset_name: The dataset name (e.g., "sensor_knowledge")
    """
    async def _async_search(query: str):
        """Internal async implementation."""
        # Search ONLY in the specified dataset
        # With ENABLE_BACKEND_ACCESS_CONTROL=true, this enforces isolation
        results = await cognee.search(
            query,
            datasets=[dataset_name]  # Critical: specify datasets parameter
        )
        
        # Format results
        if not results:
            return f"No information found in {dataset_name} for query: '{query}'"
        
        # Extract useful information from results
        formatted_results = []
        for result in results:
            if hasattr(result, 'text'):
                formatted_results.append(result.text)
            elif hasattr(result, 'content'):
                formatted_results.append(result.content)
            elif isinstance(result, dict):
                if 'text' in result:
                    formatted_results.append(result['text'])
                elif 'content' in result:
                    formatted_results.append(result['content'])
                else:
                    formatted_results.append(str(result))
            else:
                formatted_results.append(str(result))
        
        return "\n\n".join(formatted_results) if formatted_results else str(results)
    
    def agent_search(query: str):
        f"""Search the {dataset_name} knowledge base.
        
        Only searches within the {dataset_name} dataset.
        Cannot access data from other datasets due to access control.
        
        Args:
            query: The search query text
        
        Returns:
            Search results from the knowledge base
        """
        return async_to_sync(_async_search)(query=query)
    
    agent_search.__name__ = f"search_{dataset_name}"
    agent_search.__doc__ = f"Search {dataset_name} (isolated dataset only)."
    
    return agent_search


def create_agent_prune_tool(dataset_name: str):
    """
    Create a 'prune' tool for a specific dataset.
    
    Args:
        dataset_name: The dataset name
    """
    async def _async_prune():
        """Internal async implementation."""
        # Note: This will prune ALL data, not just one dataset
        # For dataset-specific pruning, you'd need to use Cognee's dataset deletion API
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        
        return f"✓ Cleared all Cognee data (affects all datasets)"
    
    def agent_prune():
        f"""Clear all data from Cognee.
        
        WARNING: This clears ALL datasets, not just {dataset_name}.
        
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
    Generate Cognee tools with proper dataset isolation.
    
    Uses Cognee's built-in access control system:
    - ENABLE_BACKEND_ACCESS_CONTROL=true enforces dataset boundaries
    - Uses Kùzu (graph) and LanceDB (vector) for isolation support
    - Each dataset is isolated at .cognee_system/databases/<user_uuid>/<dataset_uuid>.*
    
    Args:
        context_name: Context identifier (e.g., "sensor", "maintenance")
        dataset_name: Dataset name (defaults to f"{context_name}_knowledge")
        include_prune: Whether to include prune tool (default: False)
    
    Returns:
        List of FunctionTool objects with dataset isolation
    
    Example:
        sensor_tools = get_cognee_tools(context_name="sensor")
        maintenance_tools = get_cognee_tools(context_name="maintenance")
    """
    # Default dataset name
    if dataset_name is None:
        dataset_name = f"{context_name}_knowledge"
    
    print(f"   → Creating Cognee tools for dataset: {dataset_name}")
    print(f"   → Access control enabled: ENABLE_BACKEND_ACCESS_CONTROL=true")
    print(f"   → Dataset isolation enforced by Cognee")
    
    # Create tools
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
    
    print(f"   → Generated {len(tools)} tools with dataset isolation")
    
    return tools


# ==================== Convenience Functions ====================

def get_sensor_tools(include_prune: bool = False) -> List[FunctionTool]:
    """Get tools for sensor agent with dataset isolation."""
    return get_cognee_tools(
        context_name="sensor",
        dataset_name="sensor_knowledge",
        include_prune=include_prune
    )


def get_maintenance_tools(include_prune: bool = False) -> List[FunctionTool]:
    """Get tools for maintenance agent with dataset isolation."""
    return get_cognee_tools(
        context_name="maintenance",
        dataset_name="maintenance_knowledge",
        include_prune=include_prune
    )