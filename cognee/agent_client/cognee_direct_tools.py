"""
cognee_direct_tools.py

Direct Cognee integration for CAMEL agents without MCP.
"""

import asyncio
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor
from camel.toolkits import FunctionTool
import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph


class CogneeTools:
    """Direct Cognee tools for CAMEL agents."""
    
    _executor = ThreadPoolExecutor(max_workers=10)
    
    @staticmethod
    def _run_async(coro):
        """Run async function in separate thread with own event loop."""
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        
        future = CogneeTools._executor.submit(run_in_new_loop)
        return future.result(timeout=300)
    
    @staticmethod
    def search(search_query: str, search_type: str = "GRAPH_COMPLETION") -> str:
        """
        Search the Cognee knowledge graph.
        
        Args:
            search_query: Your question or search query
            search_type: Type of search - options:
                - "GRAPH_COMPLETION" (default): Natural language Q&A with full graph context
                - "RAG_COMPLETION": Traditional RAG with document chunks
                - "INSIGHTS": Entity relationships and connections
                - "CHUNKS": Raw text segments
                - "SUMMARIES": Pre-generated summaries
                - "CODE": Code-specific search
                - "FEELING_LUCKY": Auto-select best search type
        
        Returns:
            Search results as string
        """
        async def _search():
            # Handle None or empty search_type
            if not search_type or search_type.lower() == 'none':
                search_type_val = "GRAPH_COMPLETION"
            else:
                search_type_val = search_type
            
            results = await cognee.search(
                query_type=SearchType[search_type_val.upper()],
                query_text=search_query
            )
            
            if search_type_val.upper() == "CODE":
                return json.dumps(results, indent=2)
            elif search_type_val.upper() in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
                return str(results[0]) if results else "No results found"
            elif search_type_val.upper() == "INSIGHTS":
                return CogneeTools._format_insights(results)
            else:
                return str(results)
        
        try:
            return CogneeTools._run_async(_search())
        except Exception as e:
            return f"Search error: {str(e)}"
    
    @staticmethod
    def _format_insights(results):
        """Format insight results as readable relationships."""
        lines = []
        for triplet in results:
            node1, edge, node2 = triplet
            rel_type = edge.get("relationship_name", "relates_to")
            n1_name = node1.get("name", node1.get("id", "Unknown"))
            n2_name = node2.get("name", node2.get("id", "Unknown"))
            lines.append(f"{n1_name} {rel_type} {n2_name}")
        return "\n".join(lines) if lines else "No relationships found"
    
    @staticmethod
    def cognify(data: str) -> str:
        """
        Add data to Cognee knowledge graph and process it.
        
        Args:
            data: Text or information to add to knowledge graph
        
        Returns:
            Status message
        """
        async def _cognify():
            await cognee.add(data)
            await cognee.cognify(graph_model=KnowledgeGraph)
            return "✓ Data added and processed successfully"
        
        try:
            return CogneeTools._run_async(_cognify())
        except Exception as e:
            return f"Cognify error: {str(e)}"
    
    @staticmethod
    def get_status() -> str:
        """
        Check Cognee processing status.
        
        Returns:
            Current status information
        """
        async def _status():
            from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
            from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
            from cognee.modules.users.methods import get_default_user
            
            user = await get_default_user()
            dataset_id = await get_unique_dataset_id("main_dataset", user)
            status = await get_pipeline_status([dataset_id], "cognify_pipeline")
            return str(status)
        
        try:
            return CogneeTools._run_async(_status())
        except Exception as e:
            return f"Status error: {str(e)}"
    
    @staticmethod
    def list_datasets() -> str:
        """
        List all available datasets in Cognee.
        
        Returns:
            Formatted list of datasets
        """
        async def _list():
            from cognee.modules.users.methods import get_default_user
            from cognee.modules.data.methods import get_datasets
            
            user = await get_default_user()
            datasets = await get_datasets(user.id)
            
            if not datasets:
                return "No datasets found"
            
            lines = ["Available datasets:"]
            for i, ds in enumerate(datasets, 1):
                lines.append(f"{i}. {ds.name} (ID: {ds.id})")
            return "\n".join(lines)
        
        try:
            return CogneeTools._run_async(_list())
        except Exception as e:
            return f"List error: {str(e)}"
    
    @staticmethod
    def prune() -> str:
        """
        Reset Cognee knowledge graph (delete all data).
        
        Returns:
            Confirmation message
        """
        async def _prune():
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
            return "✓ Knowledge graph reset successfully"
        
        try:
            return CogneeTools._run_async(_prune())
        except Exception as e:
            return f"Prune error: {str(e)}"


def get_cognee_tools() -> List[FunctionTool]:
    """
    Get all Cognee tools as CAMEL FunctionTools.
    
    Returns:
        List of FunctionTool objects ready for ChatAgent
    
    Example:
        from cognee_direct_tools import get_cognee_tools
        
        tools = get_cognee_tools()
        agent = ChatAgent(
            system_message="You are a knowledge assistant",
            tools=tools,
            model=your_model
        )
    """
    tools = [
        FunctionTool(CogneeTools.search),
        FunctionTool(CogneeTools.cognify),
        FunctionTool(CogneeTools.get_status),
        FunctionTool(CogneeTools.list_datasets),
        FunctionTool(CogneeTools.prune),
    ]
    
    print(f"✓ Loaded {len(tools)} Cognee tools:")
    for tool in tools:
        print(f"   • {tool.get_function_name()}")
    print()
    
    return tools


# Test function
def test_cognee_tools():
    """Test the Cognee tools."""
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from camel.messages import BaseMessage
    
    # Setup model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type="gpt-oss:120b",
        url="http://10.33.205.34:11112/v1",
        model_config_dict={"temperature": 0, "max_tokens": 16384},
    )
    
    # Get tools
    cognee_tools = get_cognee_tools()
    
    # Create agent
    agent = ChatAgent(
        system_message="""You are a knowledge assistant with access to Cognee knowledge graph.
        
Available tools:
- search: Search the knowledge graph
- cognify: Add new information
- get_status: Check processing status
- list_datasets: View available datasets

Always search the knowledge graph before saying you don't know something.""",
        tools=cognee_tools,
        model=model,
    )
    
    # Test queries
    test_queries = [
        "Who is Simon Baharja Siagian",
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        msg = BaseMessage.make_user_message(role_name="User", content=query)
        response = agent.step(msg)
        print(f"Agent: {response.msg.content}")


if __name__ == "__main__":
    test_cognee_tools()