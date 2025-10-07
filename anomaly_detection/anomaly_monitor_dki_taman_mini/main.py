#!/usr/bin/env python3
"""
Weather Anomaly Monitor - Main Entry Point
Updated with CAMEL AI + MCP integration and optional agent mode
"""
import sys
import os
import asyncio
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from config import Config
from main_monitor import WeatherAnomalyMonitor
from station_agent import WeatherAnomalyCAMELAgent, ThoughtProcessLogger


async def initialize_with_agent():
    """Initialize CAMEL agent and return it"""
    
    print("\n[Agent Mode] Initializing CAMEL agent...")
    
    # Create thought logger
    thought_logger = ThoughtProcessLogger(log_file="logs/agent_thought_process.log")
    
    # Create CAMEL agent
    print("  → Creating CAMEL agent...")
    agent = WeatherAnomalyCAMELAgent(
        mcp_config_path="config/weather_anomaly.json",
        model_platform="openai",
        model_type="gpt-4-turbo",
        temperature=0.0,
        thought_logger=thought_logger
    )
    
    # Initialize agent (connects to MCP server)
    print("  → Connecting to MCP server...")
    success = await agent.initialize()
    
    if not success:
        print("\n❌ Failed to initialize agent")
        return None
    
    print("  ✓ Agent initialized successfully")
    return agent


async def run_with_agent():
    """Initialize and run the monitor with CAMEL agent"""
    
    print("=" * 60)
    print("Weather Anomaly Monitor - WITH AGENT MODE")
    print("=" * 60)
    
    # Initialize agent
    agent = await initialize_with_agent()
    
    if agent is None:
        sys.exit(1)
    
    # Create monitor with the initialized agent
    print("\n[Creating Monitor] Initializing with agent...")
    monitor = WeatherAnomalyMonitor(Config, anomaly_agent=agent)
    
    print("\n✓ System ready with AI agent!")
    print("=" * 60)
    
    # Run the monitor
    try:
        monitor.run_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup agent
        print("Cleaning up agent...")
        await agent.cleanup()
        print("✓ Cleanup complete")


def run_without_agent():
    """Run the monitor without CAMEL agent"""
    
    print("=" * 60)
    print("Weather Anomaly Monitor - WITHOUT AGENT MODE")
    print("=" * 60)
    
    # Create monitor without agent
    print("\n[Creating Monitor] Initializing without agent...")
    monitor = WeatherAnomalyMonitor(Config, anomaly_agent=None)
    
    print("\n✓ System ready (basic mode)!")
    print("=" * 60)
    
    # Run the monitor
    try:
        monitor.run_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")


def main():
    """Main entry point with argument parsing"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Weather Anomaly Monitor with optional AI agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with AI agent (default)
  python main.py
  python main.py --agent
  
  # Run without AI agent (basic mode)
  python main.py --no-agent
        """
    )
    
    parser.add_argument(
        '--agent',
        action='store_true',
        default=True,
        dest='use_agent',
        help='Run with CAMEL AI agent (default)'
    )
    
    parser.add_argument(
        '--no-agent',
        action='store_false',
        dest='use_agent',
        help='Run without AI agent (basic mode)'
    )
    
    args = parser.parse_args()
    
    # Print mode
    mode = "WITH AGENT" if args.use_agent else "WITHOUT AGENT"
    print(f"\n🚀 Starting monitor in {mode} mode\n")
    
    # Run based on mode
    if args.use_agent:
        asyncio.run(run_with_agent())
    else:
        run_without_agent()


if __name__ == "__main__":
    main()