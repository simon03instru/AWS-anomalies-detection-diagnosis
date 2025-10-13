"""
Test connection to your existing Cognee database.
This will verify you're accessing the same data as your working agent.
"""
import asyncio
import os
from mcp_client import MCPClient

# Your existing Cognee server path
EXISTING_COGNEE_SERVER = "/home/ubuntu/running/cognee/cognee/cognee-mcp/src/server.py"

# Set to match your existing database
os.environ['COGNEE_DB_PATH'] = os.getenv('COGNEE_DB_PATH', './data')
os.environ['COGNEE_DB_NAME'] = os.getenv('COGNEE_DB_NAME', 'cognee.db')


async def test_existing_database():
    """Test connection to existing Cognee database."""
    
    print("=" * 70)
    print("Testing Connection to Existing Cognee Database")
    print("=" * 70)
    print()
    
    print(f"📡 Server: {EXISTING_COGNEE_SERVER}")
    print(f"💾 Database: {os.environ['COGNEE_DB_PATH']}/{os.environ['COGNEE_DB_NAME']}")
    print()
    
    try:
        # Connect to MCP server
        print("🔌 Step 1: Connecting to MCP server...")
        mcp_client = MCPClient(EXISTING_COGNEE_SERVER)
        await mcp_client.connect()
        
        # Test list_data tool
        print("\n📊 Step 2: Testing list_data tool...")
        result = await mcp_client.call_tool("list_data", {})
        print(f"   Result: {result[:200]}...")
        
        # Test cognify_status
        print("\n📈 Step 3: Testing cognify_status tool...")
        result = await mcp_client.call_tool("cognify_status", {})
        print(f"   Status: {result[:200]}...")
        
        # Test search (if there's existing data)
        print("\n🔍 Step 4: Testing search tool...")
        result = await mcp_client.call_tool("search", {
            "query": "sensor",
            "graph_completion": True
        })
        
        if result and len(result) > 10:
            print(f"   ✅ Found existing data: {result[:200]}...")
            print("\n   🎉 SUCCESS! You're connected to existing data!")
        else:
            print(f"   ⚠️  No data found. Result: {result}")
            print("\n   💡 This is normal if the database is empty.")
            print("   Try adding data with your working CAMEL agent first.")
        
        # Cleanup
        await mcp_client.disconnect()
        
        print("\n" + "=" * 70)
        print("✅ Connection Test Complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. If you saw existing data, you're all set!")
        print("  2. If no data, add some with your working CAMEL agent")
        print("  3. Then run: python central_agent.py")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ Server script not found!")
        print(f"   Looking for: {EXISTING_COGNEE_SERVER}")
        print(f"\n   Please update EXISTING_COGNEE_SERVER with the correct path.")
        print(f"   Check where your working CAMEL agent is located.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("  1. Verify server path is correct")
        print("  2. Check database path matches your existing setup")
        print("  3. Make sure cognee is installed: pip install cognee")
        print("  4. Check .env file settings")


if __name__ == "__main__":
    print("\n🧪 Testing Existing Database Connection...\n")
    asyncio.run(test_existing_database())