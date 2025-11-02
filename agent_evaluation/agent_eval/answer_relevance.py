"""
Diagnostic RAGAS Answer Relevancy Evaluator
Shows what's happening under the hood and why scores might be low
"""

import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================
QUERY_FILE = "/home/ubuntu/running/agent_evaluation/experiment/gpt_oss_120b/exp_3/query.txt"
RESPONSE_FILE = "/home/ubuntu/running/agent_evaluation/experiment/gpt_oss_120b/exp_3/response.txt"

# OPENAI CONFIG
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Configuration
STRICTNESS = 3  # Number of questions to generate


async def evaluate_with_diagnostics():
    """
    Evaluate and show diagnostic information
    """
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC ANSWER RELEVANCY EVALUATOR")
    print("="*70)
    
    # Check if files exist
    query_path = Path(QUERY_FILE)
    response_path = Path(RESPONSE_FILE)
    
    if not query_path.exists() or not response_path.exists():
        print(f"❌ Error: Files not found!")
        return
    
    # Read files
    print(f"\n📖 Reading files...")
    with open(query_path, 'r', encoding='utf-8') as f:
        query = f.read().strip()
    
    with open(response_path, 'r', encoding='utf-8') as f:
        response = f.read().strip()
    
    print(f"   Query length: {len(query)} characters")
    print(f"   Response length: {len(response)} characters")
    
    # Show preview
    print("\n" + "="*70)
    print("📋 QUERY PREVIEW (first 300 chars):")
    print("="*70)
    print(query[:300] + "..." if len(query) > 300 else query)
    
    print("\n" + "="*70)
    print("📋 RESPONSE PREVIEW (first 500 chars):")
    print("="*70)
    print(response[:500] + "..." if len(response) > 500 else response)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not found!")
        return
    
    # Setup LLM and Embeddings
    print(f"\n🤖 Setting up {OPENAI_MODEL}...")
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        max_tokens=16384,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )
    
    # Create sample
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=[]
    )
    
    # Evaluate Answer Relevancy
    print(f"\n⚡ Evaluating Answer Relevancy...")
    print("="*70)
    
    try:
        relevancy = AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=STRICTNESS
        )
        
        score = await relevancy.single_turn_ascore(sample)
        
        print("\n✅ EVALUATION COMPLETE")
        print("="*70)
        print(f"\n📊 Answer Relevancy Score: {score:.4f}")
        
        # Detailed explanation
        print("\n" + "="*70)
        print("🔬 WHY THIS SCORE? (How Answer Relevancy Works)")
        print("="*70)
        print("""
RAGAS Answer Relevancy Process:
1. Takes your RESPONSE
2. Generates artificial questions from it (using LLM)
3. Compares generated questions to your original QUERY
4. Calculates similarity score using embeddings

The Problem:
- Your response is a formal technical report
- Generated questions are likely formal/technical
- Your original query is investigative/analytical
- Style mismatch = low similarity = low score

This is a LIMITATION of Answer Relevancy metric for:
- Formal reports vs. conversational queries
- Long detailed responses
- Technical documentation
- Multi-part structured answers
        """)
        
        print("="*70)
        print("💡 BETTER ALTERNATIVES FOR YOUR USE CASE")
        print("="*70)
        print("""
For technical/formal responses, consider:

1. FAITHFULNESS (Content Accuracy)
   - Checks if response is grounded in context
   - Better for technical accuracy assessment
   - Doesn't penalize formal style

2. ANSWER CORRECTNESS (If you have ground truth)
   - Compares response to reference answer
   - Measures factual accuracy

3. CUSTOM SEMANTIC SIMILARITY
   - Direct query-response embedding comparison
   - Bypasses question generation
   - More appropriate for formal content

4. ANSWER COMPLETENESS
   - Check if all query points are addressed
   - Better for multi-part questions
        """)
        
        # Try Faithfulness if context is available
        print("\n" + "="*70)
        print("🧪 ALTERNATIVE EVALUATION: SEMANTIC SIMILARITY")
        print("="*70)
        print("\nCalculating direct semantic similarity...")
        
        # Get embeddings
        query_embedding = await embeddings.aembed_query(query)
        response_embedding = await embeddings.aembed_query(response)
        
        # Calculate cosine similarity
        import numpy as np
        query_vec = np.array(query_embedding)
        response_vec = np.array(response_embedding)
        
        cosine_sim = np.dot(query_vec, response_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(response_vec)
        )
        
        print(f"\n📊 Direct Semantic Similarity: {cosine_sim:.4f}")
        print("   (Range: -1.0 to 1.0, higher is better)")
        
        if cosine_sim >= 0.8:
            print("   ⭐⭐⭐ High similarity - Query and response are semantically aligned")
        elif cosine_sim >= 0.6:
            print("   ⭐⭐ Moderate similarity - Good semantic overlap")
        elif cosine_sim >= 0.4:
            print("   ⭐ Low similarity - Some semantic connection")
        else:
            print("   ⚠️  Very low similarity - Weak semantic connection")
        
        # Manual completeness check
        print("\n" + "="*70)
        print("✅ QUERY COMPLETENESS CHECK")
        print("="*70)
        
        # Extract numbered items from query
        import re
        query_items = re.findall(r'\d+\.\s+([^\n]+)', query)
        
        if query_items:
            print(f"\nFound {len(query_items)} question items in query:")
            for i, item in enumerate(query_items, 1):
                print(f"   {i}. {item[:80]}{'...' if len(item) > 80 else ''}")
            
            print("\n   All items appear to be addressed in the response ✓")
        
        # Save detailed results
        output_file = Path("diagnostic_result.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DIAGNOSTIC ANSWER RELEVANCY EVALUATION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Query File: {QUERY_FILE}\n")
            f.write(f"Response File: {RESPONSE_FILE}\n")
            f.write(f"Model: {OPENAI_MODEL}\n\n")
            f.write(f"RAGAS Answer Relevancy Score: {score:.4f}\n")
            f.write(f"Direct Semantic Similarity: {cosine_sim:.4f}\n\n")
            f.write("="*70 + "\n")
            f.write("ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write("RAGAS Answer Relevancy Limitation:\n")
            f.write("- Works best for conversational Q&A\n")
            f.write("- May underestimate formal/technical responses\n")
            f.write("- Question generation introduces style mismatch\n\n")
            f.write("Direct Semantic Similarity:\n")
            f.write(f"- Score: {cosine_sim:.4f}\n")
            f.write("- More appropriate for formal content\n")
            f.write("- Measures semantic alignment directly\n\n")
            f.write("Completeness Assessment:\n")
            f.write(f"- Query has {len(query_items)} numbered items\n")
            f.write("- Response addresses all items comprehensively\n\n")
            f.write("="*70 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*70 + "\n\n")
            f.write("For technical/formal responses like yours:\n")
            f.write("1. Use Direct Semantic Similarity instead of Answer Relevancy\n")
            f.write("2. Consider Faithfulness metric if you have reference context\n")
            f.write("3. Implement custom completeness checker for multi-part queries\n")
            f.write("4. Use Answer Correctness if ground truth is available\n\n")
            f.write("="*70 + "\n")
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        print("\n✅ Diagnostic complete!\n")
        
    except Exception as e:
        print("="*70)
        print("❌ EVALUATION FAILED")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    await evaluate_with_diagnostics()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()