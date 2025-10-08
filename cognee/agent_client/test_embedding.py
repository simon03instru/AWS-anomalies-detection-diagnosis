import cognee
import asyncio

async def simple_test():
    # No prune needed - just add and process
    await cognee.add("The weather sensor detected a temperature of 25°C and humidity of 60%.")
    await cognee.cognify()
    results = await cognee.search("temperature")
    print("Results:", results)

asyncio.run(simple_test())