from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.messages import BaseMessage
import asyncio
import os
import sys
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_client import MCPClient
from tools import *
from prompt_template import *

from dotenv import load_dotenv
import os

load_dotenv()

#mcp_cognee_client = MCPClient(server_script="/home/ubuntu/running/cognee/cognee/cognee-mcp/src/server.py")
#await mcp_cognee_client.connect()

# Use local LLM
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11112/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16038,  # Use max_tokens instead of num_ctx
    },
)

##======================== Initialize weather agent ========================
weather_tool = FunctionTool(get_weather_param)


weather_agent = ChatAgent(
                system_message= WEATHER_AGENT_PROMPT,
                tools =  [weather_tool],
                model = ollama_model,
                    )

# response = weather_agent.step(
#       "What was the temperature and radiation and  pressure in latitude -16.52, longitude 13.41 on 2025-01-15 at 3:20 to 04.00 AM UTC?"
#   )
# print(response.msgs[0].content)
#print(weather_tool.get_openai_function_schema())
#print(weather_agent_prompt)


##======================== Initialize sensor agent ========================
def extract_tool_call(response_text: str):
    """Extract tool call from agent response."""
    text = response_text.strip()
    
    try:
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


async def chat_with_sensor_agent(sensor_agent, mcp_client, user_message: str):
    """Chat with sensor agent and handle tool calls."""
    
    # Send message to agent
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=user_message
    )
    
    response = sensor_agent.step(user_msg)
    agent_response = response.msg.content
    
    # Check if agent wants to call a tool
    tool_name, arguments = extract_tool_call(agent_response)
    
    if tool_name:
        print(f"   [Calling tool: {tool_name}]")
        
        # Execute the tool via MCP client
        tool_result = await mcp_client.call_tool(tool_name, arguments)
        
        # Send result back to agent for final response
        result_msg = BaseMessage.make_user_message(
            role_name="System",
            content=f"Tool result: {tool_result}"
        )
        
        final_response = sensor_agent.step(result_msg)
        return final_response.msg.content
    else:
        # No tool call, return response directly
        return agent_response


async def setup_agents():
    """
    Initialize MCP and create sensor agents.
    Returns tuple of (sensor_agent, mcp_client)
    """
    # Initialize MCP client once
    mcp_cognee_client = MCPClient("/home/ubuntu/running/diagnosis_agent/cognee/cognee-mcp/src/server.py")
    await mcp_cognee_client.connect()
    
    # Get tool descriptions
    tool_descriptions = mcp_cognee_client.get_tool_descriptions()
    
    # Create model (shared by all agents)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    
    # Create Sensor Agent
    sensor_system_message = BaseMessage.make_assistant_message(
        role_name="Sensor Agent",
        content=SENSOR_AGENT_PROMPT.format(tool_descriptions=tool_descriptions)
    )
    sensor_agent = ChatAgent(sensor_system_message, model)
    
    return sensor_agent, mcp_cognee_client

async def main():
    """Main function."""
    
    # Setup agents
    sensor_agent, mcp_client = await setup_agents()
    
    print("✅ All agents initialized!\n")
    
    try:
        # Use the helper function that handles tool calls
        answer = await chat_with_sensor_agent(
            sensor_agent,
            mcp_client,
            "list all databse you have?"
        )
        
        print(f"Sensor Agent: {answer}\n")
        
    finally:
        # Cleanup
        await mcp_client.disconnect()
        print("✅ Disconnected from MCP")


if __name__ == "__main__":
    asyncio.run(main())