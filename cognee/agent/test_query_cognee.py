import cognee
import asyncio

async def main():
    # Your question
    question = "what is the minimum operating voltage of HMP155?"
    
    print(f"\nQuery: {question}")
    print("="*60)
    
    # Search in specific dataset
    results = await cognee.search(question, datasets=["sensor_knowledge"])
    
    # Get just the search_result text
    if results and len(results) > 0:
        answer = results[0].get('search_result', ['No answer found'])[0]
        print(f"\nAnswer: {answer}\n")
    else:
        print("\nNo results found\n")

# Run
asyncio.run(main())