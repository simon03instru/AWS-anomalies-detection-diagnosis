"""
Simple CAMEL Single Agent Evaluation with Arize Phoenix
========================================================
A minimal example for evaluating a single CAMEL agent with Phoenix tracing.
"""

import os
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Phoenix imports
import phoenix as px
from phoenix.otel import register
from opentelemetry import trace

# Set your API key
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"

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

# Step 1: Launch Phoenix
print("🚀 Launching Phoenix...")
px.launch_app()

# Step 2: Register Phoenix tracing
print("📊 Registering Phoenix tracing...")
tracer_provider = register(
    project_name="simple-camel-agent",
    endpoint="http://localhost:6006/v1/traces"
)
tracer = trace.get_tracer(__name__)

# Step 3: Create a CAMEL agent
print("🤖 Creating CAMEL agent...")
system_message = BaseMessage.make_assistant_message(
    role_name="Python Expert",
    content="You are an expert Python programmer. Help users with Python coding tasks."
)

agent = ChatAgent(
    system_message=system_message,
    model= ollama_model
)

# Step 4: Run agent with tracing
print("\n💬 Starting conversation with agent...\n")

queries = [
    "Write a function to calculate fibonacci numbers",
    "Add error handling to the function",
    "Write unit tests for it"
]

for i, query in enumerate(queries):
    with tracer.start_as_current_span(f"agent_query_{i}") as span:
        # Set span attributes
        span.set_attribute("query.number", i)
        span.set_attribute("query.text", query)
        
        # Create user message
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=query
        )
        
        print(f"👤 User: {query}")
        
        # Get agent response
        response = agent.step(user_msg)
        response_text = response.msg.content
        
        # Log response
        span.set_attribute("response.text", response_text)
        span.set_attribute("response.length", len(response_text))
        
        print(f"🤖 Agent: {response_text[:200]}...\n")

print("✅ Done! View traces at http://localhost:6006")