"""
RAGAS Batch Evaluation - All Experiments with Multiple Runs
Evaluates all experiments in gpt_oss_120b, gpt_oss_20b, and gpt_4_nano folders
Runs each experiment 3 times and saves results to eval_result.txt
Includes Faithfulness and Answer Relevancy metrics with FLEXIBLE CHUNKING
MODIFIED TO USE: GPT-4o Mini
"""

import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import traceback
import re

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = "/home/ubuntu/running/agent_evaluation/experiment"
FOLDERS = ["gpt_oss_120b", "gpt_oss_20b", "gpt_4_1_mini"]
NUM_EXPERIMENTS = 20  # exp_1 to exp_20
NUM_RUNS = 2 # Run each experiment 2 times

# OPENAI CONFIG
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
# Make sure to set OPENAI_API_KEY in your .env file

# Evaluation parameters
MAX_CONTEXTS = 20
MAX_RESPONSE = 6000

# ============================================
# CHUNKING STRATEGY SELECTION
# ============================================
# Options: "semantic", "character", "sentence", "sliding_window"
CHUNKING_STRATEGY = "semantic"

# Strategy-specific parameters
SEMANTIC_CHUNK_SIZE = 500  # Target characters per chunk
CHARACTER_CHUNK_SIZE = 1000  # Characters per chunk
SENTENCES_PER_CHUNK = 5  # Sentences per chunk
SLIDING_WINDOW_SIZE = 800  # Characters per window
SLIDING_WINDOW_OVERLAP = 200  # Overlap between windows


def chunk_by_characters(text, chunk_size=1000, max_chunks=20):
    """
    Simple character-based chunking with word boundary respect.
    
    WHEN TO USE:
    - Completely unstructured text with no delimiters
    - Dense continuous text
    - When you just need equal-sized chunks
    
    PROS: Simple, predictable chunk sizes
    CONS: May split in the middle of important content
    """
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
        
        if len(chunks) >= max_chunks:
            break
    
    # Add remaining
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(' '.join(current_chunk))
    
    return chunks[:max_chunks]


def chunk_by_sentences(text, sentences_per_chunk=5, max_chunks=20):
    """
    Sentence-based chunking using regex to detect sentence boundaries.
    
    WHEN TO USE:
    - Prose, paragraphs, narrative text
    - Documentation with proper sentences
    - When semantic completeness matters
    
    PROS: Preserves complete thoughts
    CONS: Variable chunk sizes, depends on sentence length
    """
    # Split by sentence boundaries (., !, ?)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    
    return chunks[:max_chunks]


def chunk_by_sliding_window(text, window_size=800, overlap=200, max_chunks=20):
    """
    Sliding window with overlap to preserve context across boundaries.
    
    WHEN TO USE:
    - Dense technical text where context matters
    - When information might span boundaries
    - For Q&A where context is distributed
    
    PROS: No information loss at boundaries, context preserved
    CONS: Some redundancy, more chunks needed
    """
    chunks = []
    words = text.split()
    
    current_pos = 0
    
    while current_pos < len(words) and len(chunks) < max_chunks:
        # Build chunk up to window_size
        chunk_words = []
        chunk_length = 0
        
        idx = current_pos
        while idx < len(words) and chunk_length < window_size:
            word = words[idx]
            chunk_words.append(word)
            chunk_length += len(word) + 1
            idx += 1
        
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        
        # Calculate next position with overlap
        overlap_chars = 0
        overlap_words = 0
        for word in reversed(chunk_words):
            if overlap_chars + len(word) + 1 <= overlap:
                overlap_chars += len(word) + 1
                overlap_words += 1
            else:
                break
        
        current_pos = idx - overlap_words
        
        # Prevent infinite loop
        if current_pos >= len(words):
            break
        if overlap_words == 0:
            current_pos = idx
    
    return chunks[:max_chunks]


def chunk_by_semantic_breaks(text, target_chunk_size=500, max_chunks=20):
    """
    Semantic chunking that looks for natural breaks in text.
    Tries to split at: double newlines → single newlines → sentences
    
    WHEN TO USE (BEST DEFAULT):
    - Mixed content with some structure
    - Text with paragraphs or sections
    - When you want natural boundaries but no clear markers
    
    PROS: Respects natural text structure, good balance
    CONS: Variable chunk sizes
    """
    chunks = []
    
    # First try to split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # If paragraph itself is too large, split it further
        if para_size > target_chunk_size * 1.5:
            # Try splitting by single newlines
            lines = para.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_size = len(line)
                
                if current_size + line_size > target_chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
                
                if len(chunks) >= max_chunks:
                    break
        else:
            # Paragraph is reasonable size
            if current_size + para_size > target_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if len(chunks) >= max_chunks:
            break
    
    # Add remaining
    if current_chunk and len(chunks) < max_chunks:
        chunks.append('\n'.join(current_chunk))
    
    return chunks[:max_chunks]


def chunk_contexts(context_text, strategy="semantic"):
    """
    Main chunking function that routes to appropriate strategy.
    
    STRATEGY RECOMMENDATIONS:
    
    1. "semantic" (DEFAULT - BEST FOR MOST CASES)
       - Use when: Text has some structure (paragraphs, sections)
       - Example: Your weather data with Q&A sections
       
    2. "character" 
       - Use when: Completely random text with no structure
       - Example: Log files, continuous streams
       
    3. "sentence"
       - Use when: Proper prose with clear sentences
       - Example: Articles, documentation, reports
       
    4. "sliding_window"
       - Use when: Context spans boundaries, technical Q&A
       - Example: Troubleshooting guides where steps relate
    """
    if not context_text or not context_text.strip():
        return []
    
    if strategy == "character":
        return chunk_by_characters(context_text, CHARACTER_CHUNK_SIZE, MAX_CONTEXTS)
    elif strategy == "sentence":
        return chunk_by_sentences(context_text, SENTENCES_PER_CHUNK, MAX_CONTEXTS)
    elif strategy == "sliding_window":
        return chunk_by_sliding_window(context_text, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_OVERLAP, MAX_CONTEXTS)
    elif strategy == "semantic":
        return chunk_by_semantic_breaks(context_text, SEMANTIC_CHUNK_SIZE, MAX_CONTEXTS)
    else:
        # Default to semantic
        return chunk_by_semantic_breaks(context_text, SEMANTIC_CHUNK_SIZE, MAX_CONTEXTS)


async def evaluate_single_experiment(folder_name, exp_num, run_num, llm, embeddings):
    """
    Evaluate a single experiment and return the scores
    """
    exp_dir = Path(BASE_DIR) / folder_name / f"exp_{exp_num}"
    
    query_file = exp_dir / "query.txt"
    response_file = exp_dir / "response.txt"
    context_file = exp_dir / "context.txt"
    
    # Check if files exist
    if not all([query_file.exists(), response_file.exists(), context_file.exists()]):
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": "MISSING_FILES",
            "faithfulness": None,
            "relevancy": None,
            "num_contexts": 0,
            "error": "One or more files missing"
        }
    
    try:
        # Read files
        with open(query_file, 'r') as f:
            query = f.read().strip()
        
        with open(response_file, 'r') as f:
            response = f.read().strip()
        
        with open(context_file, 'r') as f:
            context_text = f.read().strip()
        
        # Chunk contexts using selected strategy
        contexts = chunk_contexts(context_text, CHUNKING_STRATEGY)
        
        # Truncate response if needed
        if len(response) > MAX_RESPONSE:
            response = response[:MAX_RESPONSE]
        
        # Create sample
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts
        )
        
        # Evaluate Faithfulness with retry
        f_score = None
        for attempt in range(3):  # Try up to 3 times
            try:
                faithfulness = Faithfulness(llm=llm)
                f_score = await faithfulness.single_turn_ascore(sample)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"      Faithfulness failed after 3 attempts: {str(e)[:100]}")
                else:
                    await asyncio.sleep(1)  # Wait before retry
        
        # Evaluate Answer Relevancy with retry
        r_score = None
        for attempt in range(3):  # Try up to 3 times
            try:
                relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
                r_score = await relevancy.single_turn_ascore(sample)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"      Relevancy failed after 3 attempts: {str(e)[:100]}")
                else:
                    await asyncio.sleep(1)  # Wait before retry
        
        status = "SUCCESS" if all([f_score is not None, r_score is not None]) else "PARTIAL"
        
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": status,
            "faithfulness": f_score,
            "relevancy": r_score,
            "num_contexts": len(contexts),
            "error": None
        }
        
    except Exception as e:
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": "ERROR",
            "faithfulness": None,
            "relevancy": None,
            "num_contexts": 0,
            "error": str(e)[:200]
        }


async def main():
    print("\n" + "="*70)
    print("🚀 RAGAS BATCH EVALUATION - GPT-4o Mini")
    print("="*70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Folders: {', '.join(FOLDERS)}")
    print(f"Experiments per folder: {NUM_EXPERIMENTS}")
    print(f"Runs per experiment: {NUM_RUNS}")
    print(f"Total evaluations: {len(FOLDERS) * NUM_EXPERIMENTS * NUM_RUNS}")
    print(f"\nLLM: {OPENAI_MODEL}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Metrics: Faithfulness, Relevancy")
    print(f"\n📊 CHUNKING STRATEGY: {CHUNKING_STRATEGY.upper()}")
    
    if CHUNKING_STRATEGY == "semantic":
        print(f"   - Target chunk size: {SEMANTIC_CHUNK_SIZE} chars")
        print(f"   - Splits at: paragraphs → lines → natural breaks")
    elif CHUNKING_STRATEGY == "character":
        print(f"   - Chunk size: {CHARACTER_CHUNK_SIZE} chars")
        print(f"   - Respects word boundaries")
    elif CHUNKING_STRATEGY == "sentence":
        print(f"   - Sentences per chunk: {SENTENCES_PER_CHUNK}")
    elif CHUNKING_STRATEGY == "sliding_window":
        print(f"   - Window size: {SLIDING_WINDOW_SIZE} chars")
        print(f"   - Overlap: {SLIDING_WINDOW_OVERLAP} chars")
    
    print(f"   - Max contexts: {MAX_CONTEXTS}")
    print("="*70 + "\n")
    
    # Verify OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    # Setup LLM and Embeddings
    print("🤖 Setting up GPT-4o Mini and embeddings...\n")
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        max_tokens=16384,  # GPT-4o mini supports up to 16,384 output tokens
        model_kwargs={
            "response_format": {"type": "json_object"}  # Force JSON output
        }
    )
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )
    
    # Store all results
    all_results = []
    
    # Track progress
    total_evals = len(FOLDERS) * NUM_EXPERIMENTS * NUM_RUNS
    current_eval = 0
    
    start_time = datetime.now()
    
    # Iterate through all folders, experiments, and runs
    for folder in FOLDERS:
        print(f"\n{'='*70}")
        print(f"📁 PROCESSING FOLDER: {folder}")
        print(f"{'='*70}\n")
        
        for exp_num in range(1, NUM_EXPERIMENTS + 1):
            print(f"  📊 Experiment {exp_num}/{NUM_EXPERIMENTS}")
            
            for run_num in range(1, NUM_RUNS + 1):
                current_eval += 1
                print(f"    🔄 Run {run_num}/{NUM_RUNS} [{current_eval}/{total_evals}]... ", end="", flush=True)
                
                result = await evaluate_single_experiment(
                    folder, exp_num, run_num, llm, embeddings
                )
                
                all_results.append(result)
                
                # Print status
                if result["status"] == "SUCCESS":
                    print(f"✅ F:{result['faithfulness']:.3f} R:{result['relevancy']:.3f} (ctx:{result['num_contexts']})")
                elif result["status"] == "PARTIAL":
                    f_str = f"{result['faithfulness']:.3f}" if result['faithfulness'] else "N/A"
                    r_str = f"{result['relevancy']:.3f}" if result['relevancy'] else "N/A"
                    print(f"⚠️  F:{f_str} R:{r_str} (ctx:{result['num_contexts']})")
                elif result["status"] == "MISSING_FILES":
                    print(f"❌ Files missing")
                else:
                    print(f"❌ Error")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # ============================================
    # SAVE RESULTS TO FILE
    # ============================================
    output_file = Path(BASE_DIR) / "eval_result_gpt4o.txt"
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING RESULTS TO: {output_file}")
    print(f"{'='*70}\n")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("="*70 + "\n")
        f.write(f"RAGAS BATCH EVALUATION RESULTS - GPT-4o Mini ({CHUNKING_STRATEGY.upper()} CHUNKING)\n")
        f.write("="*70 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)\n")
        f.write(f"LLM Model: {OPENAI_MODEL}\n")
        f.write(f"Embedding Model: {EMBEDDING_MODEL}\n")
        f.write(f"Metrics: Faithfulness, Relevancy\n")
        f.write(f"Chunking Strategy: {CHUNKING_STRATEGY}\n")
        f.write(f"Total Evaluations: {len(all_results)}\n")
        f.write("="*70 + "\n\n")
        
        # Write detailed results for each folder
        for folder in FOLDERS:
            f.write("\n" + "="*70 + "\n")
            f.write(f"FOLDER: {folder}\n")
            f.write("="*70 + "\n\n")
            
            folder_results = [r for r in all_results if r["folder"] == folder]
            
            for exp_num in range(1, NUM_EXPERIMENTS + 1):
                f.write(f"\nExperiment {exp_num}:\n")
                f.write("-" * 50 + "\n")
                
                exp_results = [r for r in folder_results if r["experiment"] == exp_num]
                
                for result in exp_results:
                    f.write(f"  Run {result['run']}: ")
                    
                    if result["status"] == "SUCCESS":
                        f.write(f"Faithfulness={result['faithfulness']:.4f}, ")
                        f.write(f"Relevancy={result['relevancy']:.4f}, ")
                        avg = (result['faithfulness'] + result['relevancy']) / 2
                        f.write(f"Average={avg:.4f} ")
                        f.write(f"(contexts={result['num_contexts']})\n")
                    elif result["status"] == "PARTIAL":
                        f_str = f"{result['faithfulness']:.4f}" if result['faithfulness'] else "N/A"
                        r_str = f"{result['relevancy']:.4f}" if result['relevancy'] else "N/A"
                        f.write(f"Faithfulness={f_str}, Relevancy={r_str} ")
                        f.write(f"[PARTIAL] (contexts={result['num_contexts']})\n")
                    else:
                        f.write(f"[{result['status']}] {result.get('error', 'Unknown error')}\n")
                
                # Calculate average for this experiment across all runs
                valid_runs = [r for r in exp_results if r["status"] == "SUCCESS"]
                if valid_runs:
                    avg_f = sum(r["faithfulness"] for r in valid_runs) / len(valid_runs)
                    avg_r = sum(r["relevancy"] for r in valid_runs) / len(valid_runs)
                    overall_avg = (avg_f + avg_r) / 2
                    avg_contexts = sum(r["num_contexts"] for r in valid_runs) / len(valid_runs)
                    f.write(f"\n  Average across {len(valid_runs)} successful runs:\n")
                    f.write(f"    Faithfulness: {avg_f:.4f}\n")
                    f.write(f"    Relevancy: {avg_r:.4f}\n")
                    f.write(f"    Overall: {overall_avg:.4f}\n")
                    f.write(f"    Avg Contexts: {avg_contexts:.1f}\n")
        
        # Write summary statistics
        f.write("\n\n" + "="*70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        for folder in FOLDERS:
            f.write(f"\n{folder}:\n")
            f.write("-" * 50 + "\n")
            
            folder_results = [r for r in all_results if r["folder"] == folder and r["status"] == "SUCCESS"]
            
            if folder_results:
                avg_f = sum(r["faithfulness"] for r in folder_results) / len(folder_results)
                avg_r = sum(r["relevancy"] for r in folder_results) / len(folder_results)
                overall_avg = (avg_f + avg_r) / 2
                avg_contexts = sum(r["num_contexts"] for r in folder_results) / len(folder_results)
                
                f.write(f"  Successful evaluations: {len(folder_results)}/{NUM_EXPERIMENTS * NUM_RUNS}\n")
                f.write(f"  Average Faithfulness: {avg_f:.4f}\n")
                f.write(f"  Average Relevancy: {avg_r:.4f}\n")
                f.write(f"  Overall Average: {overall_avg:.4f}\n")
                f.write(f"  Average Contexts per Eval: {avg_contexts:.1f}\n")
                
                # Min/Max
                f_scores = [r["faithfulness"] for r in folder_results]
                r_scores = [r["relevancy"] for r in folder_results]
                context_counts = [r["num_contexts"] for r in folder_results]
                f.write(f"  Faithfulness range: [{min(f_scores):.4f}, {max(f_scores):.4f}]\n")
                f.write(f"  Relevancy range: [{min(r_scores):.4f}, {max(r_scores):.4f}]\n")
                f.write(f"  Context count range: [{min(context_counts)}, {max(context_counts)}]\n")
            else:
                f.write(f"  No successful evaluations\n")
        
        # Overall statistics
        f.write("\n\n" + "="*70 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*70 + "\n")
        
        success_count = len([r for r in all_results if r["status"] == "SUCCESS"])
        partial_count = len([r for r in all_results if r["status"] == "PARTIAL"])
        error_count = len([r for r in all_results if r["status"] in ["ERROR", "MISSING_FILES"]])
        
        f.write(f"\nTotal evaluations: {len(all_results)}\n")
        f.write(f"  Successful: {success_count} ({success_count/len(all_results)*100:.1f}%)\n")
        f.write(f"  Partial: {partial_count} ({partial_count/len(all_results)*100:.1f}%)\n")
        f.write(f"  Failed: {error_count} ({error_count/len(all_results)*100:.1f}%)\n")
        
        if success_count > 0:
            successful_results = [r for r in all_results if r["status"] == "SUCCESS"]
            overall_f = sum(r["faithfulness"] for r in successful_results) / len(successful_results)
            overall_r = sum(r["relevancy"] for r in successful_results) / len(successful_results)
            overall_avg = (overall_f + overall_r) / 2
            overall_avg_contexts = sum(r["num_contexts"] for r in successful_results) / len(successful_results)
            
            f.write(f"\nOverall averages (successful only):\n")
            f.write(f"  Faithfulness: {overall_f:.4f}\n")
            f.write(f"  Relevancy: {overall_r:.4f}\n")
            f.write(f"  Combined: {overall_avg:.4f}\n")
            f.write(f"  Average Contexts: {overall_avg_contexts:.1f}\n")
        
        # Chunking strategy documentation
        f.write("\n" + "="*70 + "\n")
        f.write("CHUNKING STRATEGY GUIDE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Strategy Used: {CHUNKING_STRATEGY}\n\n")
        
        f.write("Available Strategies:\n\n")
        
        f.write("1. SEMANTIC (DEFAULT - RECOMMENDED)\n")
        f.write(f"   Config: target_chunk_size={SEMANTIC_CHUNK_SIZE} chars\n")
        f.write("   Best for: Text with some structure (paragraphs, sections)\n")
        f.write("   Splits at: double newlines → single newlines → natural breaks\n\n")
        
        f.write("2. CHARACTER\n")
        f.write(f"   Config: chunk_size={CHARACTER_CHUNK_SIZE} chars\n")
        f.write("   Best for: Completely unstructured/random text\n")
        f.write("   Splits at: word boundaries after N characters\n\n")
        
        f.write("3. SENTENCE\n")
        f.write(f"   Config: sentences_per_chunk={SENTENCES_PER_CHUNK}\n")
        f.write("   Best for: Prose, articles, documentation\n")
        f.write("   Splits at: sentence boundaries (. ! ?)\n\n")
        
        f.write("4. SLIDING_WINDOW\n")
        f.write(f"   Config: window_size={SLIDING_WINDOW_SIZE}, overlap={SLIDING_WINDOW_OVERLAP} chars\n")
        f.write("   Best for: Dense technical text, Q&A with distributed context\n")
        f.write("   Preserves context across boundaries with overlap\n\n")
        
        f.write(f"Max contexts per evaluation: {MAX_CONTEXTS}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"✅ Results saved to: {output_file}")
    print(f"\n📊 Quick Summary:")
    print(f"  Total evaluations: {len(all_results)}")
    print(f"  Successful: {success_count}")
    print(f"  Partial: {partial_count}")
    print(f"  Failed: {error_count}")
    print(f"  Duration: {total_duration/60:.1f} minutes")
    print("\n✅ Batch evaluation completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        traceback.print_exc()