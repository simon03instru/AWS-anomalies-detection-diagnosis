"""
CAMEL-AI Multi-Agent System with Local Ollama LLM

First, install CAMEL-AI:
pip install camel-ai

Then run this script to create role-playing agents using your local LLM.
"""

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.configs import OllamaConfig
from camel.societies import RolePlaying
from camel.messages import BaseMessage

# Configure your local Ollama LLM
def setup_local_model(model_name: str, host: str):
    """Setup CAMEL to use local Ollama model"""
    
    # Create Ollama-specific configuration
    model_config = OllamaConfig(
        temperature=0.7,
        max_tokens=2048,
    )
    
    # Create model instance pointing to your Ollama server
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=model_name,  # Your model: "gpt-oss:20b"
        model_config_dict=model_config.as_dict(),
        api_params={
            "base_url": host,  # Your Ollama server
        }
    )
    
    return model


def run_camel_agents(
    task_prompt: str,
    assistant_role: str,
    user_role: str,
    model_name: str = "gpt-oss:20b",
    host: str = "http://10.33.205.34:11112",
    max_turns: int = 10
):
    """Run CAMEL role-playing agents with local LLM"""
    
    print("=" * 70)
    print("CAMEL Multi-Agent System - Local LLM")
    print("=" * 70)
    print(f"Task: {task_prompt}")
    print(f"Assistant Role: {assistant_role}")
    print(f"User Role: {user_role}")
    print("=" * 70)
    
    # Setup the model
    model = setup_local_model(model_name, host)
    
    # Create the role-playing session
    # Set with_task_specify=False to avoid initial task specification
    # which might be causing issues
    role_play_session = RolePlaying(
        assistant_role_name=assistant_role,
        user_role_name=user_role,
        task_prompt=task_prompt,
        with_task_specify=False,  # Disable task specification for simplicity
        model=model,
        assistant_agent_kwargs=dict(model=model),
        user_agent_kwargs=dict(model=model),
    )
    
    print("\n🤖 Initializing agents...")
    
    # Get the initial message
    input_msg = role_play_session.init_chat()
    print(f"\n📋 Task:\n{input_msg.content}\n")
    
    # Run the conversation
    print("💬 Starting conversation...\n")
    print("=" * 70)
    
    chat_turn_limit = max_turns
    n = 0
    
    while n < chat_turn_limit:
        n += 1
        
        # Get assistant response
        assistant_response, user_response = role_play_session.step(input_msg)
        
        # Print assistant message
        if assistant_response.terminated:
            print(f"\n🛑 Assistant terminated the conversation.")
            break
        
        print(f"\n[Turn {n}] 🤖 {assistant_role}:")
        print("-" * 70)
        print(assistant_response.msg.content)
        
        # Print user message
        if user_response.terminated:
            print(f"\n🛑 User terminated the conversation.")
            break
        
        print(f"\n[Turn {n}] 👤 {user_role}:")
        print("-" * 70)
        print(user_response.msg.content)
        print("=" * 70)
        
        # Check for termination
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            print("\n✅ Task completed!")
            break
        
        # Update input message for next turn
        input_msg = assistant_response.msg
    
    print("\n" + "=" * 70)
    print("Conversation ended")
    print("=" * 70)


# Alternative: Using direct Ollama client (if CAMEL integration fails)
def run_simple_multi_agent(
    task_prompt: str,
    assistant_role: str,
    user_role: str,
    model_name: str = "gpt-oss:20b",
    host: str = "http://10.33.205.34:11112",
    max_turns: int = 10
):
    """Simplified multi-agent system using direct Ollama client"""
    from ollama import Client
    
    print("=" * 70)
    print("Simple Multi-Agent System - Direct Ollama")
    print("=" * 70)
    print(f"Task: {task_prompt}")
    print(f"Assistant Role: {assistant_role}")
    print(f"User Role: {user_role}")
    print("=" * 70)
    
    client = Client(host=host)
    options = {"temperature": 0.7, "num_ctx": 8192}
    
    # Initialize conversation
    assistant_history = []
    user_history = []
    
    # Create initial prompts
    assistant_sys = f"""You are a {assistant_role}. You will work on this task: {task_prompt}
Your role is to execute and provide solutions. Respond to the user's instructions and questions."""
    
    user_sys = f"""You are a {user_role}. You will guide the completion of this task: {task_prompt}
Your role is to provide instructions and feedback to the assistant. Ask questions and give directions."""
    
    # Initial user instruction
    user_msg = f"Let's work on this task: {task_prompt}. What should we do first?"
    
    print("\n💬 Starting conversation...\n")
    print("=" * 70)
    
    for turn in range(max_turns):
        # Assistant responds
        print(f"\n[Turn {turn + 1}] 🤖 {assistant_role}:")
        print("-" * 70)
        
        assistant_messages = [{"role": "system", "content": assistant_sys}]
        assistant_messages.extend(assistant_history)
        assistant_messages.append({"role": "user", "content": user_msg})
        
        assistant_response = ""
        stream = client.chat(model=model_name, messages=assistant_messages, 
                           stream=True, options=options)
        for chunk in stream:
            content = chunk['message']['content']
            assistant_response += content
            print(content, end='', flush=True)
        
        print()
        assistant_history.append({"role": "user", "content": user_msg})
        assistant_history.append({"role": "assistant", "content": assistant_response})
        
        # Check if task is done
        if "task is complete" in assistant_response.lower() or "finished" in assistant_response.lower():
            print("\n✅ Task completed!")
            break
        
        # User responds
        print(f"\n[Turn {turn + 1}] 👤 {user_role}:")
        print("-" * 70)
        
        user_messages = [{"role": "system", "content": user_sys}]
        user_messages.extend(user_history)
        user_messages.append({"role": "user", "content": f"The assistant said: {assistant_response}\n\nProvide your next instruction or feedback."})
        
        user_response = ""
        stream = client.chat(model=model_name, messages=user_messages, 
                           stream=True, options=options)
        for chunk in stream:
            content = chunk['message']['content']
            user_response += content
            print(content, end='', flush=True)
        
        print()
        print("=" * 70)
        
        user_history.append({"role": "user", "content": f"Assistant: {assistant_response}"})
        user_history.append({"role": "assistant", "content": user_response})
        
        # Update message for next turn
        user_msg = user_response
    
    print("\n" + "=" * 70)
    print("Conversation ended")
    print("=" * 70)


# Example 1: Software Development Task
def example_software_development():
    try:
        run_camel_agents(
            task_prompt="Develop a weather monitoring system with anomaly detection",
            assistant_role="Python Programmer",
            user_role="Software Architect",
            max_turns=8
        )
    except Exception as e:
        print(f"\n⚠️  CAMEL integration failed: {e}")
        print("Falling back to simple multi-agent system...\n")
        run_simple_multi_agent(
            task_prompt="Develop a weather monitoring system with anomaly detection",
            assistant_role="Python Programmer",
            user_role="Software Architect",
            max_turns=8
        )


# Example 2: Research Task
def example_research():
    try:
        run_camel_agents(
            task_prompt="Research and analyze the impact of climate change on ocean ecosystems",
            assistant_role="Marine Biologist",
            user_role="Environmental Researcher",
            max_turns=6
        )
    except Exception as e:
        print(f"\n⚠️  CAMEL integration failed: {e}")
        print("Falling back to simple multi-agent system...\n")
        run_simple_multi_agent(
            task_prompt="Research and analyze the impact of climate change on ocean ecosystems",
            assistant_role="Marine Biologist",
            user_role="Environmental Researcher",
            max_turns=6
        )


# Example 3: Business Strategy
def example_business():
    try:
        run_camel_agents(
            task_prompt="Create a marketing strategy for a new AI-powered productivity app",
            assistant_role="Marketing Strategist",
            user_role="Product Manager",
            max_turns=7
        )
    except Exception as e:
        print(f"\n⚠️  CAMEL integration failed: {e}")
        print("Falling back to simple multi-agent system...\n")
        run_simple_multi_agent(
            task_prompt="Create a marketing strategy for a new AI-powered productivity app",
            assistant_role="Marketing Strategist",
            user_role="Product Manager",
            max_turns=7
        )


if __name__ == "__main__":
    # Choose which example to run
    print("Choose an example:")
    print("1. Software Development")
    print("2. Research Task")
    print("3. Business Strategy")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        example_software_development()
    elif choice == "2":
        example_research()
    elif choice == "3":
        example_business()
    else:
        print("Invalid choice. Running software development example...")
        example_software_development()