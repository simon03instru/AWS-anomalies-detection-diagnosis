"""
cognee_direct_tools.py

Direct Cognee integration with dynamic node_set determination.
Agents autonomously decide node_sets during operation.
Uses ENABLE_BACKEND_ACCESS_CONTROL for dataset isolation.

UPDATED: Added comprehensive logging of retrieved chunks for evaluation.
"""

import cognee
from typing import Optional, List, Dict, Any
import asyncio
from functools import wraps
from camel.toolkits import FunctionTool
import nest_asyncio
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Enable dataset isolation (CRITICAL for proper isolation)
os.environ['ENABLE_BACKEND_ACCESS_CONTROL'] = 'true'
os.environ['REQUIRE_AUTHENTICATION'] = 'true'

# ==================== Logging Setup ====================

# Create logs directory if it doesn't exist
LOGS_DIR = Path("cognee_evaluation_logs")
LOGS_DIR.mkdir(exist_ok=True)

# Setup logger for chunk retrieval
chunk_logger = logging.getLogger("cognee_chunks")
chunk_logger.setLevel(logging.INFO)

# Create file handler with timestamp
log_filename = LOGS_DIR / f"chunk_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
if not chunk_logger.handlers:
    chunk_logger.addHandler(file_handler)

print(f"✓ Chunk logging initialized: {log_filename}")


def log_retrieved_chunks(
    query: str,
    node_sets: List[str],
    dataset_name: str,
    raw_results: Any,
    formatted_results: List[str],
    timestamp: str = None
) -> None:
    """
    Log retrieved chunks to JSONL file for evaluation.
    
    Args:
        query: The search query
        node_sets: Node sets searched
        dataset_name: Dataset name
        raw_results: Raw results from Cognee
        formatted_results: Formatted text results returned to agent
        timestamp: ISO timestamp (auto-generated if None)
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Extract detailed chunk information
    chunks_data = []
    
    for idx, result in enumerate(raw_results):
        chunk_info = {
            "chunk_index": idx,
            "result_type": str(type(result).__name__)
        }
        
        # Extract all available fields
        if isinstance(result, dict):
            chunk_info.update({
                "search_result": result.get("search_result"),
                "chunks": result.get("chunks"),
                "graph": result.get("graph"),
                "metadata": result.get("metadata"),
                "score": result.get("score"),
                "raw_data": result  # Store complete raw data
            })
        elif hasattr(result, '__dict__'):
            chunk_info["attributes"] = result.__dict__
        else:
            chunk_info["raw_value"] = str(result)
        
        chunks_data.append(chunk_info)
    
    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "node_sets": node_sets,
        "dataset_name": dataset_name,
        "num_results": len(raw_results),
        "chunks_data": chunks_data,
        "formatted_results": formatted_results,
        "formatted_results_length": len("\n\n".join(formatted_results)) if formatted_results else 0
    }
    
    # Log as JSON line
    chunk_logger.info(json.dumps(log_entry, default=str))
    
    print(f"   📝 Logged {len(chunks_data)} chunks for query: '{query[:50]}...'")


# ==================== Helper Functions ====================

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
    
    Returns ONLY the search_result text, but LOGS all chunks for evaluation.
    
    Args:
        dataset_name: The dataset name (e.g., "sensor_knowledge")
    """
    async def _async_search(query: str, node_sets: List[str]):
        """Internal async implementation.
        
        Args:
            query: The search query text
            node_sets: List of node sets to search within, determined by agent
        """
        timestamp = datetime.now().isoformat()
        
        # Validate node_sets
        if not node_sets:
            raise ValueError("node_sets cannot be empty. Agent must specify at least one node_set to search.")
        
        if not isinstance(node_sets, list):
            raise ValueError("node_sets must be a list of strings")
        
        # Search ONLY in the specified dataset and node_sets
        results = await cognee.search(
            query_text=query,
            node_name=node_sets,
            top_k=5  # Always return top 5 results
        )
        
        # No results
        if not results:
            node_set_str = ", ".join(node_sets)
            # Log empty result
            log_retrieved_chunks(
                query=query,
                node_sets=node_sets,
                dataset_name=dataset_name,
                raw_results=[],
                formatted_results=[],
                timestamp=timestamp
            )
            return f"No information found in {dataset_name} (node_sets: {node_set_str}) for query: '{query}'"
        
        # Extract ONLY the search_result field for agent, but log everything
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
        
        # LOG ALL CHUNKS AND RAW DATA FOR EVALUATION
        log_retrieved_chunks(
            query=query,
            node_sets=node_sets,
            dataset_name=dataset_name,
            raw_results=results,
            formatted_results=formatted_results,
            timestamp=timestamp
        )
        
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
                               
        Returns:
            Clean search results as text
            
        Raises:
            ValueError: If node_sets is empty or not a list
        
        Note:
            All retrieved chunks are logged to cognee_evaluation_logs/ for evaluation.
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
    - ALL RETRIEVED CHUNKS ARE LOGGED FOR EVALUATION
    
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
        # All search results are logged to cognee_evaluation_logs/
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
    print(f"   → ALL CHUNKS LOGGED TO: {LOGS_DIR}")
    
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
    print(f"   → Chunk logging enabled for evaluation")
    
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


# ==================== Evaluation Utilities ====================

def load_evaluation_logs(log_file: str = None) -> List[Dict[str, Any]]:
    """
    Load chunk retrieval logs for evaluation.
    
    Args:
        log_file: Specific log file to load (default: latest)
    
    Returns:
        List of log entries as dictionaries
    """
    if log_file is None:
        # Find latest log file
        log_files = sorted(LOGS_DIR.glob("chunk_retrieval_*.jsonl"))
        if not log_files:
            return []
        log_file = log_files[-1]
    else:
        log_file = Path(log_file)
    
    entries = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    return entries


def print_evaluation_summary(log_file: str = None):
    """
    Print summary statistics of chunk retrieval logs.
    
    Args:
        log_file: Specific log file to analyze (default: latest)
    """
    entries = load_evaluation_logs(log_file)
    
    if not entries:
        print("No evaluation logs found.")
        return
    
    print("\n" + "="*60)
    print("CHUNK RETRIEVAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Total queries: {len(entries)}")
    print(f"Total chunks retrieved: {sum(e['num_results'] for e in entries)}")
    print(f"Average chunks per query: {sum(e['num_results'] for e in entries) / len(entries):.2f}")
    print(f"Queries with no results: {sum(1 for e in entries if e['num_results'] == 0)}")
    
    # Dataset breakdown
    datasets = {}
    for entry in entries:
        ds = entry['dataset_name']
        datasets[ds] = datasets.get(ds, 0) + 1
    
    print("\nQueries by dataset:")
    for ds, count in datasets.items():
        print(f"  - {ds}: {count}")
    
    # Node set usage
    node_sets_used = set()
    for entry in entries:
        node_sets_used.update(entry['node_sets'])
    
    print(f"\nUnique node sets used: {len(node_sets_used)}")
    print(f"Node sets: {', '.join(sorted(node_sets_used))}")
    
    print("="*60 + "\n")