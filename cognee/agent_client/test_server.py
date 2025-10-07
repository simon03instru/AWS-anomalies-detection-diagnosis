import httpx

async def test_sse():
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', 'http://localhost:8000/sse') as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                print(f"Received: {line}")

import asyncio
asyncio.run(test_sse())