import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Handles MCP server connection and tool management."""
    
    def __init__(self, server_script: str = "cognee_server.py"):
        self.server_script = server_script
        self.session = None
        self.stdio_context = None
        self.mcp_tools = []
        
    async def connect(self):
        """Connect to MCP server and retrieve available tools."""
        print("Connecting to MCP Server...")
        
        if not os.path.exists(self.server_script):
            raise FileNotFoundError(f"Server script not found: {self.server_script}")
        
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script, "--transport", "stdio"],
            env=os.environ.copy()
        )
        
        self.stdio_context = stdio_client(server_params)
        read, write = await self.stdio_context.__aenter__()
        
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        print("Connected to MCP Server")
        
        # Get available tools
        tools_list = await self.session.list_tools()
        self.mcp_tools = tools_list.tools
        
        print(f"Found {len(self.mcp_tools)} MCP tools:")
        for tool in self.mcp_tools:
            print(f"   - {tool.name}")
        print()
        
        return self.mcp_tools
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        print("\nDisconnecting from MCP Server...")
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning during disconnect: {e}")
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the result."""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            response_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    response_text += content.text
            return response_text or "Operation completed successfully."
        except Exception as e:
            return f"Error calling tool: {str(e)}"
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for system message."""
        return "\n".join([
            f"- {tool.name}: {tool.description or 'No description'}"
            for tool in self.mcp_tools
        ])