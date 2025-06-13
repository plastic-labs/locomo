import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
import time  # Add for timing
from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
from task_eval.evaluation import rule_based_eval_question_answering, llm_as_judge_eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers
from task_eval.claude_utils import get_claude_answers
from task_eval.gemini_utils import get_gemini_answers
from task_eval.hf_llm_utils import init_hf_model, get_hf_answers
from task_eval.honcho_utils import get_honcho_answers

import numpy as np
import google.generativeai as genai

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--scoring-modes', nargs='+', default=['f1'], choices=['f1', 'llm'],
                        help='Space-separated list of scoring modes to run (f1, llm)')
    parser.add_argument('--override-cached-scores', action='store_true',
                        help='Force re-grading by LLM even if cached scores exist')
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    print(f"[DEBUG] Script started at {time.strftime('%H:%M:%S')}")

    # get arguments
    args = parse_args()
    print(f"[DEBUG] Arguments parsed successfully")
    print(f"[DEBUG] Model: {args.model}")
    print(f"[DEBUG] Data file: {args.data_file}")
    print(f"[DEBUG] Output file: {args.out_file}")
    print(f"[DEBUG] Batch size: {args.batch_size}")

    print("******************  Evaluating Model %s ***************" % args.model)

    # Model-specific initialization
    print(f"[DEBUG] Starting model-specific initialization...")
    init_start = time.time()
    
    if 'gpt' in args.model:
        # set openai API key
        print(f"[DEBUG] Setting up OpenAI API key...")
        set_openai_key()
        print(f"[DEBUG] OpenAI API key set successfully")

    elif 'claude' in args.model:
        # set anthropic API key
        print(f"[DEBUG] Setting up Anthropic API key...")
        set_anthropic_key()
        print(f"[DEBUG] Anthropic API key set successfully")

    elif 'gemini' in args.model:
        # set gemini API key
        print(f"[DEBUG] Setting up Gemini API key...")
        set_gemini_key()
        if args.model == "gemini-pro-1.0":
            model_name = "models/gemini-1.0-pro-latest"

        gemini_model = genai.GenerativeModel(model_name)
        print(f"[DEBUG] Gemini model initialized successfully")
    
    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        print(f"[DEBUG] Initializing HuggingFace model...")
        hf_pipeline, hf_model_name = init_hf_model(args)
        print(f"[DEBUG] HuggingFace model initialized: {hf_model_name}")
    
    elif args.model == 'honcho':
        # Honcho uses its own API key from environment
        print(f"[DEBUG] Using Honcho for evaluation")
        print(f"[DEBUG] Honcho environment variables:")
        print(f"[DEBUG]   HONCHO_BASE_URL: {os.environ.get('HONCHO_BASE_URL', 'Not set')}")
        print(f"[DEBUG]   HONCHO_ENVIRONMENT: {os.environ.get('HONCHO_ENVIRONMENT', 'Not set')}")

    else:
        raise NotImplementedError

    init_time = time.time() - init_start
    print(f"[DEBUG] Model initialization completed in {init_time:.2f} seconds")

    # load conversations
    print(f"[DEBUG] Loading conversations from {args.data_file}...")
    load_start = time.time()
    samples = json.load(open(args.data_file))
    load_time = time.time() - load_start
    print(f"[DEBUG] Loaded {len(samples)} samples in {load_time:.2f} seconds")
    
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    print(f"[DEBUG] Prediction key: {prediction_key}")
    print(f"[DEBUG] Model key: {model_key}")
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"[DEBUG] Created output directory: {out_dir}")
    
    # load the output file if it exists to check for overwriting
    print(f"[DEBUG] Checking for existing output file...")
    if os.path.exists(args.out_file):
        try:
            with open(args.out_file, 'r') as f:
                existing_data = json.load(f)
            
            # Handle different possible formats
            if isinstance(existing_data, list):
                # Expected format: list of samples
                out_samples = {d['sample_id']: d for d in existing_data if isinstance(d, dict) and 'sample_id' in d}
                print(f"[DEBUG] Found existing output file with {len(out_samples)} samples")
                
                # Safety check: count total existing predictions
                total_existing_predictions = 0
                for sample_data in existing_data:
                    if 'qa' in sample_data:
                        total_existing_predictions += sum(1 for qa in sample_data['qa'] if prediction_key in qa)
                
                if total_existing_predictions > 50 and not args.overwrite:
                    print(f"[SAFETY WARNING] Found {total_existing_predictions} existing predictions!")
                    print(f"[SAFETY WARNING] This represents substantial progress. Use --overwrite flag if you really want to replace these.")
                    print(f"[SAFETY WARNING] Otherwise, the script will resume from where it left off.")
                    
            elif isinstance(existing_data, dict) and 'sample_id' in existing_data:
                # Single sample format (from progress saving)
                out_samples = {existing_data['sample_id']: existing_data}
                print(f"[DEBUG] Found existing progress file with 1 sample")
            else:
                print(f"[DEBUG] Existing output file has unexpected format, starting fresh")
                out_samples = {}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[DEBUG] Error reading existing output file: {e}. Starting fresh.")
            out_samples = {}
    else:
        out_samples = {}
        print(f"[DEBUG] No existing output file found")

    print(f"[DEBUG] Processing samples...")
    for sample_idx, data in enumerate(samples):
        sample_start = time.time()
        print(f"[DEBUG] Processing sample {sample_idx + 1}/{len(samples)}: {data['sample_id']}")
        print(f"[DEBUG] Available existing samples: {list(out_samples.keys())}")
        print(f"[DEBUG] Looking for sample_id: '{data['sample_id']}'")
        print(f"[DEBUG] Sample found in existing results: {data['sample_id'] in out_samples}")

        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            # Load existing QA data but preserve original structure
            existing_qa = out_samples[data['sample_id']]['qa'].copy()
            out_data['qa'] = data['qa'].copy()  # Start with original data structure
            
            print(f"[DEBUG] Merging {len(existing_qa)} existing QA items with {len(data['qa'])} input QA items")
            
            # Merge in any existing predictions while preserving original fields
            predictions_merged = 0
            for i, original_qa in enumerate(out_data['qa']):
                if i < len(existing_qa) and prediction_key in existing_qa[i]:
                    out_data['qa'][i][prediction_key] = existing_qa[i][prediction_key]
                    predictions_merged += 1
                    # Copy any other prediction-related fields
                    for key in existing_qa[i]:
                        if key.endswith('_prediction') or key.endswith('_context') or key.endswith('_f1') or key.endswith('_recall'):
                            out_data['qa'][i][key] = existing_qa[i][key]
            
            print(f"[DEBUG] Merged {predictions_merged} existing predictions")
            
            # Count how many questions already have predictions
            existing_predictions = sum(1 for qa in out_data['qa'] if prediction_key in qa)
            print(f"[DEBUG] Resuming sample {data['sample_id']} with {existing_predictions}/{len(out_data['qa'])} questions already processed")
        else:
            out_data['qa'] = data['qa'].copy()
            print(f"[DEBUG] Using fresh QA data for sample {data['sample_id']} ({len(data['qa'])} questions)")

        print(f"[DEBUG] Calling model-specific answer function...")
        answer_start = time.time()
        
        if 'gpt' in args.model:
            # get answers for each sample
            answers = get_gpt_answers(data, out_data, prediction_key, args)
        elif 'claude' in args.model:
            answers = get_claude_answers(data, out_data, prediction_key, args)
        elif 'gemini' in args.model:
            answers = get_gemini_answers(gemini_model, data, out_data, prediction_key, args)
        elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
            answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name)
        elif args.model == 'honcho':
            answers = get_honcho_answers(data, out_data, prediction_key, args)
        else:
            raise NotImplementedError

        answer_time = time.time() - answer_start
        print(f"[DEBUG] Got answers in {answer_time:.2f} seconds")

        # evaluate individual QA samples and save the score
        print(f"[DEBUG] Evaluating question-answering results using modes {args.scoring_modes} ...")
        eval_start = time.time()

        all_exact = []
        # Run F1 scoring if requested
        if 'f1' in args.scoring_modes:
            print("[DEBUG] Computing rule-based F1 scores ...")
            exact_f1, _, _ = rule_based_eval_question_answering(answers['qa'], prediction_key)
            all_exact.append(('f1', exact_f1))

        # Run LLM judge if requested
        if 'llm' in args.scoring_modes:
            print("[DEBUG] Computing LLM-as-judge scores ...")
            set_openai_key()
            exact_llm, _, _ = llm_as_judge_eval_question_answering(
                answers['qa'], prediction_key, model_key,
                override_cached=args.override_cached_scores
            )
            all_exact.append(('llm', exact_llm))

        # For backwards compatibility, keep `exact_matches` equal to first mode selected
        exact_matches = all_exact[0][1]
        lengths = 0.0
        recall = [1]*len(exact_matches)
        eval_time = time.time() - eval_start
        print(f"[DEBUG] Evaluation completed in {eval_time:.2f} seconds")
        
        print(f"[DEBUG] exact_matches: {exact_matches}")
        print(f"[DEBUG] length of exact_matches: {len(exact_matches)}")
        print(f"[DEBUG] lengths: {lengths}")
        print(f"[DEBUG] recall: {recall}")
        print(f"[DEBUG] length of recall: {len(recall)}")
        # Map the returned scores back onto the original QA list.
        # `exact_matches` contains a score ONLY for questions that were actually
        # processed (i.e., those that had a prediction). We walk through the QA
        # list in order and assign scores sequentially to those same items so
        # indices stay aligned even if some questions were skipped.
        score_idx = 0
        for i, qa in enumerate(answers['qa']):
            if prediction_key in qa:
                if score_idx >= len(exact_matches):
                    break  # safety – should not happen
                qa[model_key + '_f1'] = round(exact_matches[score_idx], 3)
                if args.use_rag and len(recall) > score_idx:
                    qa[model_key + '_recall'] = round(recall[score_idx], 3)
                score_idx += 1

        out_samples[data['sample_id']] = answers
        
        # Persist progress after each sample so we can resume safely
        try:
            with open(args.out_file, 'w') as f:
                json.dump(list(out_samples.values()), f, indent=2)
        except Exception as e:
            print(f"[DEBUG] Failed to write interim results for sample {data['sample_id']}: {e}")
        
        sample_time = time.time() - sample_start
        print(f"[DEBUG] Sample {data['sample_id']} completed in {sample_time:.2f} seconds")

    print(f"[DEBUG] Running aggregate accuracy analysis...")
    analysis_start = time.time()

    if 'f1' in args.scoring_modes:
        analyze_aggr_acc(
            args.data_file, args.out_file,
            args.out_file.replace('.json', '_f1_stats.json'),
            model_key, model_key + '_f1', rag=args.use_rag
        )
    if 'llm' in args.scoring_modes:
        analyze_aggr_acc(
            args.data_file, args.out_file,
            args.out_file.replace('.json', '_llm_stats.json'),
            model_key, model_key + '_llm', rag=args.use_rag
        )

    analysis_time = time.time() - analysis_start
    print(f"[DEBUG] Analysis completed in {analysis_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"[DEBUG] Script completed successfully in {total_time:.2f} seconds total")

main()

