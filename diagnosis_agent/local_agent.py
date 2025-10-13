from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# Use your remote Ollama server
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11112/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 4096,  # Use max_tokens instead of num_ctx
    },
)

# Create agent and test
agent = ChatAgent("You are a helpful assistant.", model=ollama_model)
response = agent.step("Solve the math problem step by step: What is 35 + 27 * 8?")
print(response.msg.content)