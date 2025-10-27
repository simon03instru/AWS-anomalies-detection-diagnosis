"""
RAGAS Batch Evaluation - All Experiments with Multiple Runs
Evaluates all experiments in gpt_oss_120b, gpt_oss_20b, and gpt_4_nano folders
Runs each experiment 3 times and saves results to eval_result.txt
"""

import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import traceback

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = "/home/ubuntu/running/agent_evalaution/experiment"
FOLDERS = ["gpt_oss_120b", "gpt_oss_20b", "gpt_4_nano"]
NUM_EXPERIMENTS = 7  # exp_1 to exp_7
NUM_RUNS = 3  # Run each experiment 3 times

# LOCAL LLM SERVER CONFIG
LOCAL_LLM_HOST = "http://10.33.205.34:11112"
LOCAL_LLM_MODEL = "gpt-oss:120b"
EMBEDDING_MODEL = "nomic-embed-text"

# Evaluation parameters
LINES_PER_CHUNK = 40
MAX_CONTEXTS = 5
MAX_RESPONSE = 5000


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
            all_contexts = [line.strip() for line in context_text.split('\n') if line.strip()]
        
        # Process contexts
        contexts = []
        for i in range(0, len(all_contexts), LINES_PER_CHUNK):
            chunk = all_contexts[i:i+LINES_PER_CHUNK]
            contexts.append(" ".join(chunk))
        contexts = contexts[:MAX_CONTEXTS]
        
        # Truncate response if needed
        if len(response) > MAX_RESPONSE:
            response = response[:MAX_RESPONSE]
        
        # Create sample
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts
        )
        
        # Evaluate Faithfulness
        f_score = None
        try:
            faithfulness = Faithfulness(llm=llm)
            f_score = await faithfulness.single_turn_ascore(sample)
        except Exception as e:
            print(f"      Faithfulness failed: {str(e)[:100]}")
        
        # Evaluate Answer Relevancy
        r_score = None
        try:
            relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
            r_score = await relevancy.single_turn_ascore(sample)
        except Exception as e:
            print(f"      Relevancy failed: {str(e)[:100]}")
        
        status = "SUCCESS" if (f_score is not None and r_score is not None) else "PARTIAL"
        
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": status,
            "faithfulness": f_score,
            "relevancy": r_score,
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
            "error": str(e)[:200]
        }


async def main():
    print("\n" + "="*70)
    print("🚀 RAGAS BATCH EVALUATION - ALL EXPERIMENTS")
    print("="*70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Folders: {', '.join(FOLDERS)}")
    print(f"Experiments per folder: {NUM_EXPERIMENTS}")
    print(f"Runs per experiment: {NUM_RUNS}")
    print(f"Total evaluations: {len(FOLDERS) * NUM_EXPERIMENTS * NUM_RUNS}")
    print(f"\nLLM: {LOCAL_LLM_MODEL}")
    print(f"Host: {LOCAL_LLM_HOST}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print("="*70 + "\n")
    
    # Setup LLM and Embeddings (reuse for all evaluations)
    print("🤖 Setting up LLM and embeddings...\n")
    llm = OllamaLLM(
        model=LOCAL_LLM_MODEL,
        base_url=LOCAL_LLM_HOST,
        temperature=0,
        num_ctx=16000,
    )
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LOCAL_LLM_HOST
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
                    print(f"✅ F:{result['faithfulness']:.3f} R:{result['relevancy']:.3f}")
                elif result["status"] == "PARTIAL":
                    f_str = f"{result['faithfulness']:.3f}" if result['faithfulness'] else "N/A"
                    r_str = f"{result['relevancy']:.3f}" if result['relevancy'] else "N/A"
                    print(f"⚠️  F:{f_str} R:{r_str}")
                elif result["status"] == "MISSING_FILES":
                    print(f"❌ Files missing")
                else:
                    print(f"❌ Error")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # ============================================
    # SAVE RESULTS TO FILE
    # ============================================
    output_file = Path(BASE_DIR) / "eval_result.txt"
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING RESULTS TO: {output_file}")
    print(f"{'='*70}\n")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("="*70 + "\n")
        f.write("RAGAS BATCH EVALUATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)\n")
        f.write(f"LLM Model: {LOCAL_LLM_MODEL}\n")
        f.write(f"Embedding Model: {EMBEDDING_MODEL}\n")
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
                        f.write(f"Average={avg:.4f}\n")
                    elif result["status"] == "PARTIAL":
                        f_str = f"{result['faithfulness']:.4f}" if result['faithfulness'] else "N/A"
                        r_str = f"{result['relevancy']:.4f}" if result['relevancy'] else "N/A"
                        f.write(f"Faithfulness={f_str}, Relevancy={r_str} [PARTIAL]\n")
                    else:
                        f.write(f"[{result['status']}] {result.get('error', 'Unknown error')}\n")
                
                # Calculate average for this experiment across all runs
                valid_runs = [r for r in exp_results if r["status"] == "SUCCESS"]
                if valid_runs:
                    avg_f = sum(r["faithfulness"] for r in valid_runs) / len(valid_runs)
                    avg_r = sum(r["relevancy"] for r in valid_runs) / len(valid_runs)
                    overall_avg = (avg_f + avg_r) / 2
                    f.write(f"\n  Average across {len(valid_runs)} successful runs:\n")
                    f.write(f"    Faithfulness: {avg_f:.4f}\n")
                    f.write(f"    Relevancy: {avg_r:.4f}\n")
                    f.write(f"    Overall: {overall_avg:.4f}\n")
        
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
                
                f.write(f"  Successful evaluations: {len(folder_results)}/{NUM_EXPERIMENTS * NUM_RUNS}\n")
                f.write(f"  Average Faithfulness: {avg_f:.4f}\n")
                f.write(f"  Average Relevancy: {avg_r:.4f}\n")
                f.write(f"  Overall Average: {overall_avg:.4f}\n")
                
                # Min/Max
                f_scores = [r["faithfulness"] for r in folder_results]
                r_scores = [r["relevancy"] for r in folder_results]
                f.write(f"  Faithfulness range: [{min(f_scores):.4f}, {max(f_scores):.4f}]\n")
                f.write(f"  Relevancy range: [{min(r_scores):.4f}, {max(r_scores):.4f}]\n")
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
            
            f.write(f"\nOverall averages (successful only):\n")
            f.write(f"  Faithfulness: {overall_f:.4f}\n")
            f.write(f"  Relevancy: {overall_r:.4f}\n")
            f.write(f"  Combined: {overall_avg:.4f}\n")
        
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