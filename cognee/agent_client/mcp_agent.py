"""
CAMEL Cognee Agent - Uses direct MCP connection (your working method)
with CAMEL's agent system
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
from camel.types import ModelPlatformType

# Use local LLM
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11112/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16384,
    },
)

class SimpleCogneeAgent:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session = None
        self.stdio_context = None
        self.agent = None
        self.tools = []
        
    async def connect(self):
        """Connect using your working method."""
        print("Connecting to Cognee MCP Server...")
        
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
        
        tools_list = await self.session.list_tools()
        self.tools = tools_list.tools
        
        print(f"Connected! Found {len(self.tools)} tools:")
        for tool in self.tools:
            print(f"   - {tool.name}")
        print()
    
    async def disconnect(self):
        """Disconnect."""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)
    
    def setup_agent(self):
        """Setup CAMEL agent."""
        tool_desc = "\n".join([
            f"- {t.name}: {t.description or 'No description'}"
            for t in self.tools
        ])
        
        system_msg = f"""You are a helpful AI assistant with Cognee knowledge graph access.

Available tools:
{tool_desc}

When using a tool, respond with JSON:
{{
    "tool": "tool_name",
    "arguments": {{"param": "value"}}
}}

ALWAYS USE SEARCH FOR QUESTIONS.
IF NO INFO FOUND, SAY SO."""
        
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Assistant",
                content=system_msg
            ),
            model=ollama_model,
        )
        print("Agent ready\n")
    
    async def call_tool(self, tool_name: str, args: dict) -> str:
        """Call MCP tool."""
        try:
            result = await self.session.call_tool(tool_name, arguments=args)
            return result.content[0].text if result.content else "Done"
        except Exception as e:
            return f"Error: {e}"
    
    def _extract_tool_call(self, text: str):
        """Extract tool call from response."""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                if 'tool' in data:
                    return data['tool'], data.get('arguments', {})
        except:
            pass
        return None, None
    
    async def chat(self, message: str) -> str:
        """Chat with agent."""
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=message
        )
        
        response = self.agent.step(user_msg)
        text = response.msg.content
        
        tool_name, args = self._extract_tool_call(text)
        
        if tool_name:
            print(f"   [Using: {tool_name}]")
            result = await self.call_tool(tool_name, args)
            
            result_msg = BaseMessage.make_user_message(
                role_name="System",
                content=f"Tool result: {result}"
            )
            
            final = self.agent.step(result_msg)
            return final.msg.content
        
        return text
    
    async def run(self):
        """Interactive mode."""
        print("=" * 60)
        print("CAMEL-Cognee Assistant")
        print("=" * 60)
        print("\nCommands: /add /file /status /list /quit")
        print("=" * 60)
        print()
        
        while True:
            try:
                inp = input("You: ").strip()
                
                if not inp:
                    continue
                
                if inp == "/quit":
                    break
                
                elif inp == "/add":
                    print("\nEnter text (Ctrl+D to finish):")
                    lines = []
                    try:
                        while True:
                            lines.append(input())
                    except EOFError:
                        pass
                    text = "\n".join(lines)
                    if text.strip():
                        result = await self.call_tool("cognify", {"data": text})
                        print(f"\n{result}\n")
                
                elif inp == "/file":
                    path = input("File path: ").strip()
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            result = await self.call_tool("cognify", {"data": f.read()})
                        print(f"\n{result}\n")
                    else:
                        print("File not found\n")
                
                elif inp == "/status":
                    result = await self.call_tool("cognify_status", {})
                    print(f"\n{result}\n")
                
                elif inp == "/list":
                    result = await self.call_tool("list_data", {})
                    print(f"\n{result}\n")
                
                else:
                    response = await self.chat(inp)
                    print(f"\nAssistant: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nBye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


async def main():
    server_path = os.getenv(
        "COGNEE_MCP_SERVER",
        "/home/ubuntu/running/cognee/cognee/cognee-mcp/src/server.py"
    )
    
    if not os.path.exists(server_path):
        print(f"Server not found: {server_path}")
        print("Set COGNEE_MCP_SERVER environment variable")
        return
    
    print(f"Using server: {server_path}\n")
    
    agent = SimpleCogneeAgent(server_path)
    
    try:
        await agent.connect()
        agent.setup_agent()
        await agent.run()
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())