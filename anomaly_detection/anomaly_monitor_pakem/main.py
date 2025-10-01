#!/usr/bin/env python3
"""
Weather Anomaly Monitor - Main Entry Point
Updated with CAMEL AI + MCP integration
"""
import sys
import os
import asyncio

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from config import Config
from main_monitor import WeatherAnomalyMonitor
from station_agent import WeatherAnomalyCAMELAgent, ThoughtProcessLogger


async def initialize_and_run():
    """Initialize CAMEL agent and run the monitor"""
    
    print("=" * 60)
    print("Initializing Weather Anomaly Monitor")
    print("=" * 60)
    
    # Create thought logger
    thought_logger = ThoughtProcessLogger(log_file="logs/agent_thought_process.log")
    
    # Create CAMEL agent
    print("\n[1/3] Creating CAMEL agent...")
    agent = WeatherAnomalyCAMELAgent(
        mcp_config_path="config/weather_anomaly.json",
        model_platform="gemini",
        model_type="gemini-2.0-flash-exp",
        temperature=0.0,
        thought_logger=thought_logger
    )
    
    # Initialize agent (connects to MCP server)
    print("[2/3] Initializing agent (connecting to MCP server)...")
    success = await agent.initialize()
    
    if not success:
        print("\n❌ Failed to initialize agent")
        sys.exit(1)
    
    print("✓ Agent initialized successfully")
    
    # Create monitor with the initialized agent
    print("[3/3] Creating monitor...")
    monitor = WeatherAnomalyMonitor(Config, anomaly_agent=agent)
    
    print("\n✓ System ready!")
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


def main():
    """Main entry point"""
    asyncio.run(initialize_and_run())


if __name__ == "__main__":
    main()