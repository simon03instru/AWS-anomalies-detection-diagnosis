"""
DeepEval Batch Evaluation - Simplified Version with Reason Logging
Evaluates: Answer Relevancy, Faithfulness, Contextual Relevancy (overall score only)
No Contextual Precision (requires ground truth)
Single chunk for Contextual Relevancy (overall score, no detailed insights)
Includes evaluation reasons for debugging and analysis
"""

from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import traceback

from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
from deepeval.metrics.faithfulness import FaithfulnessTemplate

load_dotenv()


# Evaluatioin Prompt 
from deepeval.metrics import FaithfulnessMetric


# Define custom template
class CustomFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_claims(actual_output: str):
        return f"""Based on the given text, please extract a comprehensive list of facts that can inferred from the provided text.
            CONTEXT INTERPRETATION INSTRUCTIONS:

            This document contains weather data from satellites, APIs, and sensor systems.
            Data in JSON, CSV, or structured formats represents ACTUAL MEASURED VALUES from
            real sensors and observation systems, not examples or placeholders.

            CRITICAL RULES:
            ✓ JSON field "temperature": 25.3 means temperature IS 25.3°C (stated fact)
            ✓ JSON field "humidity": 0 means humidity IS 0% (stated fact)  
            ✓ All numeric values are real measurements that can be cited
            ✓ Timestamps show when observations were recorded
            ✓ Treat structured data fields as explicit factual statements

            DO NOT claim information is "not mentioned" if it exists in JSON/structured format.
            All data values can be referenced as stated facts in your evaluation.

            Example:
            Example Text:
            "CNN claims that the sun is 3 times smaller than earth."

            Example JSON:
            {{
                "claims": []
            }}
            ===== END OF EXAMPLE ======

            Text:
            {actual_output}

            JSON:
            """


class CustomContextualRelevency(ContextualRelevancyTemplate):
    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""Based on the input and context, please generate a JSON object to indicate whether each statement found in the context is relevant to the provided input.
        CONTEXT INTERPRETATION INSTRUCTIONS:

            This document contains weather data from satellites, APIs, and sensor systems.
            Data in JSON, CSV, or structured formats represents ACTUAL MEASURED VALUES from
            real sensors and observation systems, not examples or placeholders.

            CRITICAL RULES:
            ✓ JSON field "temperature": 25.3 means temperature IS 25.3°C (stated fact)
            ✓ JSON field "humidity": 0 means humidity IS 0% (stated fact)  
            ✓ All numeric values are real measurements that can be cited
            ✓ Timestamps show when observations were recorded
            ✓ Treat structured data fields as explicit factual statements
        
        This information is always relevant because the input queries always pertain to weather data.


        Example JSON:
        {{
            "verdicts": [
                {{
                    "verdict": "yes",
                    "statement": "...",
                }}
            ]
        }}
        **

        Input:
        {input}

        Context:
        {context}

        JSON:
        """


# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = "/home/ubuntu/running/agent_evaluation/experiment"
FOLDERS = ["gpt_oss_120b", "gpt_oss_20b", "gpt_4_1_mini"]
NUM_EXPERIMENTS = 20  # exp_1 to exp_20
NUM_RUNS = 1  # Run each experiment 1 time

# MODEL CONFIGURATION
MODEL = "gpt-4o-mini"

# Thresholds
ANSWER_RELEVANCY_THRESHOLD = 1
FAITHFULNESS_THRESHOLD = 1
CONTEXTUAL_RELEVANCY_THRESHOLD = 1

# Context handling
MAX_CONTEXT_LENGTH = 10000  # Truncate context if too long
USE_CONTEXT = True  # Set to False to skip context-based metrics


def evaluate_single_experiment(folder_name, exp_num, run_num):
    """
    Evaluate a single experiment and return the scores WITH REASONS
    Uses single chunk for overall Contextual Relevancy score
    """
    exp_dir = Path(BASE_DIR) / folder_name / f"exp_{exp_num}"
    
    query_file = exp_dir / "query.txt"
    response_file = exp_dir / "response.txt"
    context_file = exp_dir / "context.txt"
    
    # Check if files exist
    if not all([query_file.exists(), response_file.exists()]):
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": "MISSING_FILES",
            "answer_relevancy": None,
            "faithfulness": None,
            "contextual_relevancy": None,
            "answer_relevancy_reason": None,
            "faithfulness_reason": None,
            "contextual_relevancy_reason": None,
            "error": "One or more required files missing"
        }
    
    try:
        # Read files
        with open(query_file, 'r', encoding='utf-8') as f:
            query = f.read().strip()
        
        with open(response_file, 'r', encoding='utf-8') as f:
            response = f.read().strip()
        
        # Read context if available - USE AS SINGLE CHUNK
        retrieval_context = []
        if USE_CONTEXT and context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                context_text = f.read().strip()
                # Return entire context as one chunk (truncate if too long)
                if context_text:
                    retrieval_context = [context_text[:MAX_CONTEXT_LENGTH]]
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=retrieval_context
        )
        
        # Evaluate Answer Relevancy
        relevancy_score = None
        relevancy_reason = None
        try:
            answer_relevancy = AnswerRelevancyMetric(
                threshold=ANSWER_RELEVANCY_THRESHOLD,
                model=MODEL,
                #verbose_mode = True,
                include_reason=True  # Include reason for logging
            )
            answer_relevancy.measure(test_case)
            relevancy_score = answer_relevancy.score
            relevancy_reason = answer_relevancy.reason if hasattr(answer_relevancy, 'reason') else "No reason provided"
        except Exception as e:
            print(f"      Answer Relevancy failed: {str(e)[:100]}")
            relevancy_reason = f"ERROR: {str(e)[:200]}"
        
        # Evaluate Faithfulness (only if context available)
        faithfulness_score = None
        faithfulness_reason = None
        if USE_CONTEXT and retrieval_context:
            try:
                faithfulness = FaithfulnessMetric(
                    evaluation_template=CustomFaithfulnessTemplate,
                    threshold=FAITHFULNESS_THRESHOLD,
                    model=MODEL,
                    #verbose_mode = True,
                    include_reason=True  # Include reason for logging
                )
                faithfulness.measure(test_case)
                faithfulness_score = faithfulness.score
                faithfulness_reason = faithfulness.reason if hasattr(faithfulness, 'reason') else "No reason provided"
            except Exception as e:
                print(f"      Faithfulness failed: {str(e)[:100]}")
                faithfulness_reason = f"ERROR: {str(e)[:200]}"
        
        # Evaluate Contextual Relevancy (only if context available)
        # Uses single chunk for overall score
        contextual_relevancy_score = None
        contextual_relevancy_reason = None
        if USE_CONTEXT and retrieval_context:
            try:
                contextual_relevancy = ContextualRelevancyMetric(
                    evaluation_template=CustomContextualRelevency,
                    threshold=CONTEXTUAL_RELEVANCY_THRESHOLD,
                    model=MODEL,
                    #verbose_mode = True,
                    include_reason=True  # Include reason for logging
                )
                contextual_relevancy.measure(test_case)
                contextual_relevancy_score = contextual_relevancy.score
                contextual_relevancy_reason = contextual_relevancy.reason if hasattr(contextual_relevancy, 'reason') else "No reason provided"
            except Exception as e:
                print(f"      Contextual Relevancy failed: {str(e)[:100]}")
                contextual_relevancy_reason = f"ERROR: {str(e)[:200]}"
        
        status = "SUCCESS" if relevancy_score is not None else "PARTIAL"
        
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": status,
            "answer_relevancy": relevancy_score,
            "faithfulness": faithfulness_score,
            "contextual_relevancy": contextual_relevancy_score,
            "answer_relevancy_reason": relevancy_reason,
            "faithfulness_reason": faithfulness_reason,
            "contextual_relevancy_reason": contextual_relevancy_reason,
            "num_contexts": len(retrieval_context),
            "error": None
        }
        
    except Exception as e:
        return {
            "folder": folder_name,
            "experiment": exp_num,
            "run": run_num,
            "status": "ERROR",
            "answer_relevancy": None,
            "faithfulness": None,
            "contextual_relevancy": None,
            "answer_relevancy_reason": None,
            "faithfulness_reason": None,
            "contextual_relevancy_reason": None,
            "num_contexts": 0,
            "error": str(e)[:200]
        }


def main():
    print("\n" + "="*70)
    print("🚀 DEEPEVAL BATCH EVALUATION - WITH REASON LOGGING")
    print("="*70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Folders: {', '.join(FOLDERS)}")
    print(f"Experiments per folder: {NUM_EXPERIMENTS}")
    print(f"Runs per experiment: {NUM_RUNS}")
    print(f"Total evaluations: {len(FOLDERS) * NUM_EXPERIMENTS * NUM_RUNS}")
    print(f"\nModel: {MODEL}")
    print(f"Metrics: Answer Relevancy, Faithfulness, Contextual Relevancy (overall)")
    print(f"Note: Using single chunk for Contextual Relevancy (overall score)")
    print(f"Note: Contextual Precision excluded (requires ground truth)")
    print(f"Note: Including evaluation reasons for analysis")
    print(f"Use Context: {USE_CONTEXT}")
    print("="*70 + "\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        return
    
    # Store all results
    all_results = []
    
    # Track progress
    total_evals = len(FOLDERS) * NUM_EXPERIMENTS * NUM_RUNS
    current_eval = 0
    
    start_time = datetime.now()
    
    # Iterate EXPERIMENT-BY-EXPERIMENT across all models
    # First do exp_1 for all models, then exp_2 for all models, etc.
    for exp_num in range(1, NUM_EXPERIMENTS + 1):
        print(f"\n{'='*70}")
        print(f"📊 EXPERIMENT {exp_num}/{NUM_EXPERIMENTS}")
        print(f"{'='*70}\n")
        
        for folder in FOLDERS:
            print(f"  📁 Model: {folder}")
            
            for run_num in range(1, NUM_RUNS + 1):
                current_eval += 1
                print(f"    🔄 Run {run_num}/{NUM_RUNS} [{current_eval}/{total_evals}]... ", end="", flush=True)
                
                result = evaluate_single_experiment(folder, exp_num, run_num)
                all_results.append(result)
                
                # Print status
                if result["status"] == "SUCCESS":
                    ar = result['answer_relevancy']
                    f = result['faithfulness']
                    cr = result['contextual_relevancy']
                    
                    ar_str = f"AR:{ar:.3f}" if ar is not None else "AR:N/A"
                    f_str = f"F:{f:.3f}" if f is not None else "F:N/A"
                    cr_str = f"CR:{cr:.3f}" if cr is not None else "CR:N/A"
                    
                    print(f"✅ {ar_str} {f_str} {cr_str} (ctx:{result['num_contexts']})")
                elif result["status"] == "PARTIAL":
                    print(f"⚠️  Partial success")
                elif result["status"] == "MISSING_FILES":
                    print(f"❌ Files missing")
                else:
                    print(f"❌ Error")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # ============================================
    # SAVE RESULTS TO FILE
    # ============================================
    output_file = Path(BASE_DIR) / "eval_result_deepeval_with_reasons.txt"
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING RESULTS TO: {output_file}")
    print(f"{'='*70}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*70 + "\n")
        f.write("DEEPEVAL BATCH EVALUATION RESULTS (WITH REASONS)\n")
        f.write("="*70 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Metrics: Answer Relevancy, Faithfulness, Contextual Relevancy (overall)\n")
        f.write(f"Note: Contextual Relevancy uses single chunk (overall score only)\n")
        f.write(f"Note: Contextual Precision excluded (requires ground truth)\n")
        f.write(f"Note: Includes evaluation reasons for each metric\n")
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
                f.write("-" * 70 + "\n")
                
                exp_results = [r for r in folder_results if r["experiment"] == exp_num]
                
                for result in exp_results:
                    f.write(f"\n  Run {result['run']}:\n")
                    
                    if result["status"] == "SUCCESS":
                        ar = result['answer_relevancy']
                        fth = result['faithfulness']
                        cr = result['contextual_relevancy']
                        
                        ar_str = f"{ar:.4f}" if ar is not None else "N/A"
                        fth_str = f"{fth:.4f}" if fth is not None else "N/A"
                        cr_str = f"{cr:.4f}" if cr is not None else "N/A"
                        
                        f.write(f"    Scores: AR={ar_str}, F={fth_str}, CR={cr_str}")
                        
                        # Calculate average of available scores
                        scores = [s for s in [ar, fth, cr] if s is not None]
                        if scores:
                            avg = sum(scores) / len(scores)
                            f.write(f", Avg={avg:.4f}")
                        
                        f.write(f" (contexts={result['num_contexts']})\n\n")
                        
                        # Write reasons
                        if result.get('answer_relevancy_reason'):
                            f.write(f"    📝 Answer Relevancy Reason:\n")
                            f.write(f"       {result['answer_relevancy_reason']}\n\n")
                        
                        if result.get('faithfulness_reason'):
                            f.write(f"    📝 Faithfulness Reason:\n")
                            f.write(f"       {result['faithfulness_reason']}\n\n")
                        
                        if result.get('contextual_relevancy_reason'):
                            f.write(f"    📝 Contextual Relevancy Reason:\n")
                            f.write(f"       {result['contextual_relevancy_reason']}\n\n")
                        
                    else:
                        f.write(f"    [{result['status']}] {result.get('error', 'Unknown error')}\n")
                
                # Calculate average for this experiment
                valid_runs = [r for r in exp_results if r["status"] == "SUCCESS"]
                if valid_runs:
                    ar_scores = [r['answer_relevancy'] for r in valid_runs if r['answer_relevancy'] is not None]
                    f_scores = [r['faithfulness'] for r in valid_runs if r['faithfulness'] is not None]
                    cr_scores = [r['contextual_relevancy'] for r in valid_runs if r['contextual_relevancy'] is not None]
                    
                    f.write(f"\n  Averages across runs:\n")
                    if ar_scores:
                        f.write(f"    Answer Relevancy: {sum(ar_scores)/len(ar_scores):.4f}\n")
                    if f_scores:
                        f.write(f"    Faithfulness: {sum(f_scores)/len(f_scores):.4f}\n")
                    if cr_scores:
                        f.write(f"    Contextual Relevancy: {sum(cr_scores)/len(cr_scores):.4f}\n")
                
                f.write("\n")
        
        # Write summary statistics
        f.write("\n\n" + "="*70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        for folder in FOLDERS:
            f.write(f"\n{folder}:\n")
            f.write("-" * 50 + "\n")
            
            folder_results = [r for r in all_results if r["folder"] == folder and r["status"] == "SUCCESS"]
            
            if folder_results:
                ar_scores = [r['answer_relevancy'] for r in folder_results if r['answer_relevancy'] is not None]
                f_scores = [r['faithfulness'] for r in folder_results if r['faithfulness'] is not None]
                cr_scores = [r['contextual_relevancy'] for r in folder_results if r['contextual_relevancy'] is not None]
                
                f.write(f"  Successful evaluations: {len(folder_results)}/{NUM_EXPERIMENTS * NUM_RUNS}\n\n")
                
                if ar_scores:
                    avg_ar = sum(ar_scores) / len(ar_scores)
                    f.write(f"  Answer Relevancy:\n")
                    f.write(f"    Average: {avg_ar:.4f}\n")
                    f.write(f"    Range: [{min(ar_scores):.4f}, {max(ar_scores):.4f}]\n\n")
                
                if f_scores:
                    avg_f = sum(f_scores) / len(f_scores)
                    f.write(f"  Faithfulness:\n")
                    f.write(f"    Average: {avg_f:.4f}\n")
                    f.write(f"    Range: [{min(f_scores):.4f}, {max(f_scores):.4f}]\n\n")
                
                if cr_scores:
                    avg_cr = sum(cr_scores) / len(cr_scores)
                    f.write(f"  Contextual Relevancy (Overall):\n")
                    f.write(f"    Average: {avg_cr:.4f}\n")
                    f.write(f"    Range: [{min(cr_scores):.4f}, {max(cr_scores):.4f}]\n\n")
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
    print(f"\n💡 TIP: Check the output file for detailed reasons behind each score!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        traceback.print_exc()
