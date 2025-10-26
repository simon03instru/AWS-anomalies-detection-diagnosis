"""
RAGAS Test with Local Ollama Server - Updated Version
Uses langchain-ollama (not deprecated) and works around JSON issues
"""

import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    # ============================================
    # ADJUST PATH
    # ============================================
    EXPERIMENT_DIR = "/home/ubuntu/running/agent_evalaution/eval_data/gpt_4o_mini/exp_2"
    
    QUERY_FILE = f"{EXPERIMENT_DIR}/query.txt"
    RESPONSE_FILE = f"{EXPERIMENT_DIR}/response.txt"
    CONTEXT_FILE = f"{EXPERIMENT_DIR}/context.txt"
    
    # ============================================
    # LOCAL LLM SERVER CONFIG
    # ============================================
    LOCAL_LLM_HOST = "http://10.33.205.34:11112"
    LOCAL_LLM_MODEL = "gpt-oss:120b"
    EMBEDDING_MODEL = "nomic-embed-text"  # Change if you have different model
    
    # ============================================
    # READ FILES
    # ============================================
    print("📂 Reading files...\n")
    
    with open(QUERY_FILE, 'r') as f:
        query = f.read().strip()
    
    with open(RESPONSE_FILE, 'r') as f:
        response = f.read().strip()
    
    with open(CONTEXT_FILE, 'r') as f:
        context_text = f.read().strip()
        all_contexts = [line.strip() for line in context_text.split('\n') if line.strip()]
    
    print(f"✓ Original: {len(all_contexts)} context lines")
    
    # Combine even MORE aggressively for local LLM (slower)
    LINES_PER_CHUNK = 40  # Bigger chunks = fewer contexts = faster
    contexts = []
    for i in range(0, len(all_contexts), LINES_PER_CHUNK):
        chunk = all_contexts[i:i+LINES_PER_CHUNK]
        contexts.append(" ".join(chunk))
    
    # Limit to 5 contexts max for local LLM (faster evaluation)
    contexts = contexts[:5]
    
    # Truncate response more aggressively for local LLM
    MAX_RESPONSE = 5000  # Shorter = faster and more reliable
    original_response_len = len(response)
    if len(response) > MAX_RESPONSE:
        response = response[:MAX_RESPONSE]
    
    print(f"✓ Combined: {len(contexts)} context chunks")
    print(f"✓ Query: {len(query)} chars")
    print(f"✓ Response: {len(response)} chars (original: {original_response_len})\n")
    
    # ============================================
    # SETUP LOCAL LLM (Updated imports)
    # ============================================
    print(f"🤖 Setting up local LLM evaluator...")
    print(f"   Host: {LOCAL_LLM_HOST}")
    print(f"   Model: {LOCAL_LLM_MODEL}")
    print(f"   Embedding: {EMBEDDING_MODEL}\n")
    
    # Use updated langchain-ollama package
    llm = OllamaLLM(
        model=LOCAL_LLM_MODEL,
        base_url=LOCAL_LLM_HOST,
        temperature=0,
        num_ctx=16000,
        # Don't force JSON format - let RAGAS handle it
    )
    
    # Use updated OllamaEmbeddings
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LOCAL_LLM_HOST
    )
    
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=contexts
    )
    
    # ============================================
    # EVALUATE BOTH METRICS
    # ============================================
    f_score = None
    r_score = None
    
    # Evaluate Faithfulness
    print("⏳ Evaluating Faithfulness (may take 5-10 minutes)...\n")
    try:
        faithfulness = Faithfulness(llm=llm)
        f_score = await faithfulness.single_turn_ascore(sample)
        print(f"   ✅ Faithfulness Score: {f_score:.4f}\n")
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Faithfulness failed: {error_msg[:250]}")
        
        if "EOF" in error_msg or "Invalid JSON" in error_msg:
            print(f"      → Your LLM is not returning valid JSON")
            print(f"      → This is common with some models")
            print(f"      → Try using OpenAI for evaluation instead\n")
        else:
            print(f"      → Check if server is accessible and model exists\n")
    
    # Evaluate Answer Relevancy
    print("⏳ Evaluating Answer Relevancy (may take 3-5 minutes)...\n")
    try:
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        r_score = await relevancy.single_turn_ascore(sample)
        print(f"   ✅ Answer Relevancy Score: {r_score:.4f}\n")
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Answer Relevancy failed: {error_msg[:250]}")
        
        if "not found" in error_msg.lower() or "embedding" in error_msg.lower():
            print(f"      → Embedding model '{EMBEDDING_MODEL}' not found")
            print(f"      → Install with: ollama pull {EMBEDDING_MODEL}\n")
        elif "EOF" in error_msg or "Invalid JSON" in error_msg:
            print(f"      → Your LLM is not returning valid JSON")
            print(f"      → Try using OpenAI for evaluation instead\n")
        else:
            print(f"      → Check server connection and model availability\n")
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    print("="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    if f_score is not None:
        print(f"Faithfulness Score:     {f_score:.4f}")
    else:
        print(f"Faithfulness Score:     FAILED")
    
    if r_score is not None:
        print(f"Answer Relevancy Score: {r_score:.4f}")
    else:
        print(f"Answer Relevancy Score: FAILED")
    
    if f_score is not None and r_score is not None:
        avg_score = (f_score + r_score) / 2
        print(f"Average Score:          {avg_score:.4f}")
        print("="*60)
        
        print("\n💡 Interpretation:")
        if avg_score >= 0.9:
            print("   🟢 Excellent! Response is grounded and relevant.")
        elif avg_score >= 0.7:
            print("   🟡 Good! Minor improvements possible.")
        elif avg_score >= 0.5:
            print("   🟠 Fair. Some issues detected.")
        else:
            print("   🔴 Poor. Significant problems with response.")
    else:
        print("="*60)
        print("\n⚠️  Evaluation failed with local LLM.")
        print("\n💡 Recommendations:")
        print("   1. Use OpenAI for evaluation (complete_test.py)")
        print("      - Much faster and more reliable")
        print("      - Small cost (~$0.01 per experiment)")
        print()
        print("   2. Keep using local LLM for agent responses")
        print("      - Free and private")
        print("      - Just use OpenAI for evaluation only")
        print()
        print("   3. Hybrid approach is best:")
        print("      - Local LLM: Generate agent responses (free)")
        print("      - OpenAI: Evaluate responses (fast, reliable)")
    
    print("\n✅ Evaluation completed!")
    return {"faithfulness": f_score, "relevancy": r_score}

if __name__ == "__main__":
    print("\n🚀 RAGAS Evaluation Test with Local Ollama LLM")
    print("="*60 + "\n")
    
    print("⚠️  Note: Local LLM evaluation is experimental and may not work")
    print("   with all models. If it fails, use OpenAI for evaluation.\n")
    
    try:
        result = asyncio.run(main())
        print(f"\n📊 Final Scores: {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()