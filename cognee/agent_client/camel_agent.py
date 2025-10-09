"""
CAMEL AI Agent with Direct MCP Tool Access

Uses CAMEL's ChatAgent for conversation but bypasses its tool system
to directly call async MCP tools, avoiding sync/async conversion overhead.

Prerequisites:
- pip install camel-ai mcp
- Environment: OPENAI_API_KEY

Usage:
    python camel_cognee_agent.py
"""

import asyncio
import os
import sys
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Use local LLM
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11112/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16384,  # Use max_tokens instead of num_ctx
    },
)

class CAMELCogneeAgent:
    """CAMEL agent with direct async MCP tool access."""
    
    def __init__(self, server_script: str = "cognee_server.py"):
        self.server_script = server_script
        self.session = None
        self.stdio_context = None
        self.agent = None
        self.mcp_tools = []
        
    async def connect(self):
        """Connect to Cognee MCP server."""
        print("Connecting to Cognee MCP Server...")
        
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
        
        print("Connected to Cognee MCP Server")
        
        # Get available tools
        tools_list = await self.session.list_tools()
        self.mcp_tools = tools_list.tools
        
        print(f"Found {len(self.mcp_tools)} MCP tools:")
        for tool in self.mcp_tools:
            print(f"   - {tool.name}")
        print()
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        print("\nDisconnecting...")
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: {e}")
    
    def setup_agent(self):
        """Setup CAMEL agent (without tools in CAMEL's system)."""
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
        )
        
        # Build tool descriptions for system message
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description or 'No description'}"
            for tool in self.mcp_tools
        ])
        
        system_message = f"""You are a helpful AI assistant with access to Cognee knowledge graph through MCP tools.

                            Available MCP tools:
                            {tool_descriptions}

                            IMPORTANT: When you need to use a tool, respond with a JSON object in this EXACT format:
                            {{
                                "tool": "tool_name",
                                "arguments": {{"param": "value"}}
                            }}
                            IF NOT SPECIFIED, LEAVE ARGUMENTS AS DEFAULT,
                            ALWAYS USE SEARCH FOR ANSWERING A QUESTION, ALWAYS USE GRAPH_COMPLETION FOR SEARCH¸
                            IF THE TOOLS RETURN NO RELATED INFORMATION, JUST SAY SO."""
        
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Cognee Assistant",
                content=system_message
            ),
            model=ollama_model,
        )
        
        print(f"CAMEL Agent initialized\n")
    
    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool directly."""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            response_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    response_text += content.text
            return response_text or "Operation completed successfully."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _extract_tool_call(self, text: str) -> tuple:
        """Extract tool call from agent response."""
        text = text.strip()
        
        # Look for JSON tool call
        try:
            # Try to find JSON in the response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                tool_call = json.loads(json_str)
                if 'tool' in tool_call:
                    return tool_call.get('tool'), tool_call.get('arguments', {})
        except json.JSONDecodeError:
            pass
        
        return None, None
    
    async def chat(self, user_message: str) -> str:
        """Chat with CAMEL agent and handle tool calls."""
        # Send message to CAMEL agent
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=user_message
        )
        
        response = self.agent.step(user_msg)
        agent_response = response.msg.content
        
        # Check if agent wants to call a tool
        tool_name, arguments = self._extract_tool_call(agent_response)
        
        if tool_name:
            print(f"   [Calling MCP tool: {tool_name}]")
            
            # Execute the tool
            tool_result = await self.call_mcp_tool(tool_name, arguments)
            
            # Send result back to agent for final response
            result_msg = BaseMessage.make_user_message(
                role_name="System",
                content=f"Tool result: {tool_result}"
            )
            
            final_response = self.agent.step(result_msg)
            return final_response.msg.content
        else:
            # No tool call, return response directly
            return agent_response
    
    async def add_content_helper(self, content: str):
        """Helper to add content directly."""
        result = await self.session.call_tool("cognify", arguments={"data": content})
        print(f"\n{result.content[0].text}")
        print("Processing in background... Use /status to check.\n")
    
    async def interactive_mode(self):
        """Run interactive chat."""
        print("=" * 70)
        print("CAMEL-Cognee AI Assistant - Interactive Mode")
        print("=" * 70)
        print("\nQuick Commands:")
        print("  /add    - Add text to knowledge graph")
        print("  /file   - Add file content")
        print("  /status - Check processing status")
        print("  /list   - List all datasets")
        print("  /quit   - Exit")
        print("\nOr chat naturally - I'll use MCP tools as needed!")
        print("=" * 70)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    break
                
                elif user_input == "/add":
                    print("\nEnter content (Ctrl+D when done):")
                    print("-" * 70)
                    lines = []
                    try:
                        while True:
                            lines.append(input())
                    except EOFError:
                        pass
                    content = "\n".join(lines)
                    if content.strip():
                        await self.add_content_helper(content)
                
                elif user_input == "/file":
                    filepath = input("Enter file path: ").strip()
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        await self.add_content_helper(content)
                    else:
                        print(f"File not found: {filepath}\n")
                
                elif user_input == "/status":
                    result = await self.session.call_tool("cognify_status", arguments={})
                    print(f"\nStatus:\n{result.content[0].text}\n")
                
                elif user_input == "/list":
                    result = await self.session.call_tool("list_data", arguments={})
                    print(f"\nDatasets:\n{result.content[0].text}\n")
                
                else:
                    # Chat with agent
                    response = await self.chat(user_input)
                    print(f"\nAssistant: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return
    
    server_script = os.getenv("COGNEE_MCP_SERVER", "/home/ubuntu/running/cognee/cognee/cognee-mcp/src/server.py")
    print(f"Using MCP server: {server_script}\n")
    
    agent = CAMELCogneeAgent(server_script)
    
    try:
        await agent.connect()
        agent.setup_agent()
        await agent.interactive_mode()
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        await agent.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")