import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import os
import re
import time  # Add for timing
from tqdm import tqdm
from honcho import Honcho
from typing import Dict, List, Any
from dotenv import load_dotenv
import random

load_dotenv()

# Initialize Honcho client
HONCHO_BASE_URL = os.environ.get("HONCHO_BASE_URL", "http://localhost:8000")
HONCHO_ENVIRONMENT = os.environ.get("HONCHO_ENVIRONMENT", "local")

# Load mappings
MAPPINGS_FILE = "data/honcho_mappings.json"

# Rate limiting configuration
MIN_REQUEST_INTERVAL = 2.5  # Minimum seconds between requests (24 requests/minute)
last_request_time = 0

# Question answering prompts
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. Do not include explanations or other text.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {}

Respond only with the letter (a or b) that corresponds to the correct answer.
Answer:
"""

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

"""

# ------------------------------------------------------------------
# Helper for Category-5 (adversarial) answer mapping
# ------------------------------------------------------------------

def get_cat_5_answer(model_prediction: str, answer_key: dict[str, str]) -> str:
    """Convert model's letter choice back to the full answer string.

    1. If prediction is just 'a' or 'b' (possibly with parentheses) return
       the mapped text.
    2. Otherwise return the prediction unchanged (model wrote full answer).
    """
    text = model_prediction.strip().lower()
    if text in {"a", "b"}:
        return answer_key.get(text, model_prediction)
    if text in {"(a)", "(b)"}:
        return answer_key.get(text.strip("()"), model_prediction)
    return model_prediction

def throttle_request():
    """Apply throttling to stay under rate limits."""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        print(f"[THROTTLE] Waiting {sleep_time:.1f}s to maintain rate limit...")
        time.sleep(sleep_time)
    
    last_request_time = time.time()


def load_honcho_mappings():
    """Load the mappings created during ingestion."""
    print(f"[HONCHO DEBUG] Loading mappings from {MAPPINGS_FILE}...")
    start_time = time.time()
    
    if not os.path.exists(MAPPINGS_FILE):
        raise FileNotFoundError(f"Mappings file not found: {MAPPINGS_FILE}. Run ingest_locomo_to_honcho.py first.")
    
    with open(MAPPINGS_FILE, 'r') as f:
        mappings = json.load(f)
    
    load_time = time.time() - start_time
    print(f"[HONCHO DEBUG] Loaded {len(mappings)} mappings in {load_time:.2f} seconds")
    return mappings


def identify_target_user(question: str, speakers: Dict[str, str]) -> tuple[str, str]:
    """
    Identify which user the question is about.
    Returns (user_name, user_id)
    """
    question_lower = question.lower()
    
    # Check each speaker name in the question
    for speaker_name, user_id in speakers.items():
        if speaker_name.lower() in question_lower:
            print(f"[HONCHO DEBUG] Question targets speaker: {speaker_name}")
            return speaker_name, user_id
    
    # Default to first speaker if no specific speaker mentioned
    # This handles general questions like "What happened on X date?"
    first_speaker = list(speakers.keys())[0]
    print(f"[HONCHO DEBUG] No specific speaker mentioned, defaulting to: {first_speaker}")
    return first_speaker, speakers[first_speaker]


def get_session_for_question(honcho: Honcho, app_id: str, user_id: str, question: str) -> str:
    """
    Determine which session to query based on the question.
    For now, we'll use a simple approach and let the dialectic API handle cross-session retrieval.
    """
    print(f"[HONCHO DEBUG] Getting sessions for user {user_id}...")
    start_time = time.time()
    
    # List all sessions for the user
    sessions = list(honcho.apps.users.sessions.list(
        app_id=app_id,
        user_id=user_id,
        is_active=True
    ))
    
    list_time = time.time() - start_time
    print(f"[HONCHO DEBUG] Found {len(sessions)} sessions in {list_time:.2f} seconds")
    
    if not sessions:
        raise ValueError(f"No sessions found for user {user_id}")
    
    # For questions with temporal context, we could parse dates
    # For now, return the first session and let dialectic handle cross-session
    # The dialectic API will search across all sessions anyway
    session_id = sessions[0].id
    print(f"[HONCHO DEBUG] Using session: {session_id}")
    return session_id


def process_single_question(honcho: Honcho, app_id: str, speakers: Dict[str, str], question: str, category: int) -> str:
    """Process a single question using Honcho's dialectic API."""
    print(f"[HONCHO DEBUG] Processing question (category {category}): {question[:100]}...")
    start_time = time.time()
    
    # Identify target user
    identify_start = time.time()
    speaker_name, user_id = identify_target_user(question, speakers)
    identify_time = time.time() - identify_start
    print(f"[HONCHO DEBUG] User identification took {identify_time:.2f} seconds")
    
    # Get a session for the user (dialectic will search across all sessions)
    session_start = time.time()
    session_id = get_session_for_question(honcho, app_id, user_id, question)
    session_time = time.time() - session_start
    print(f"[HONCHO DEBUG] Session lookup took {session_time:.2f} seconds")
    
    # Use appropriate prompt based on category
    if category == 5:
        # Adversarial questions
        prompt = QA_PROMPT_CAT_5.format(question)
        print(f"[HONCHO DEBUG] Using adversarial prompt for category 5")
    else:
        prompt = QA_PROMPT.format(question)
        print(f"[HONCHO DEBUG] Using standard prompt for category {category}")
    
    # Retry logic for dialectic API calls
    max_retries = 5
    base_delay = 2.0
    
    for attempt in range(max_retries + 1):
        try:
            # Apply throttling before making the request
            throttle_request()
            
            print(f"[HONCHO DEBUG] Calling dialectic API (attempt {attempt + 1}/{max_retries + 1})...")
            api_start = time.time()
            
            response = honcho.apps.users.sessions.chat(
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                queries=prompt
            )
            
            api_time = time.time() - api_start
            print(f"[HONCHO DEBUG] Dialectic API call completed in {api_time:.2f} seconds")
            
            # Extract answer from response
            if response and hasattr(response, 'content') and response.content:
                answer = response.content.strip()
            else:
                print(f"[HONCHO DEBUG] Empty or invalid response from dialectic API")
                answer = None
            
            if not answer or answer.lower() in ['none', 'null', '']:
                print(f"[HONCHO DEBUG] Received empty/null answer: '{answer}'")
                # For adversarial questions (category 5), return appropriate default
                if category == 5:
                    answer = "Empty response"
                else:
                    answer = "No answer found"
            else:
                print(f"[HONCHO DEBUG] Raw answer: {answer[:200]}...")
                
                # Clean up the answer
                if answer.lower().startswith("based on"):
                    # Remove preamble if present
                    lines = answer.split('\n')
                    answer = lines[-1].strip() if lines else answer
                    print(f"[HONCHO DEBUG] Cleaned answer: {answer[:200]}...")
            
            total_time = time.time() - start_time
            print(f"[HONCHO DEBUG] Question processed in {total_time:.2f} seconds total")
            return answer
        
        except Exception as e:
            error_str = str(e)
            print(f"[HONCHO DEBUG] Error calling dialectic API: {error_str}")
            
            # Check if it's a rate limit error (429) or server error (500 which might be rate limit)
            is_rate_limit_error = False
            suggested_wait_time = None
            
            if "rate limit" in error_str.lower() or "429" in error_str:
                is_rate_limit_error = True
                print(f"[HONCHO DEBUG] Rate limit detected in error: {error_str}")
            elif "500" in error_str or "internal server error" in error_str.lower():
                # 500 errors from dialectic API are often caused by underlying rate limits
                is_rate_limit_error = True
                print(f"[HONCHO DEBUG] Server error detected (likely rate limit): {error_str}")
            
            # Try to extract retry-after information
            if "try again in" in error_str.lower():
                match = re.search(r'try again in (\d+(?:\.\d+)?)\s*([sm])', error_str.lower())
                if match:
                    value, unit = match.groups()
                    suggested_wait_time = float(value) * (60 if unit == 'm' else 1)
                    print(f"[HONCHO DEBUG] Suggested wait time: {suggested_wait_time} seconds")
            
            # If it's not a rate limit error or we've exhausted retries, give up
            if not is_rate_limit_error or attempt == max_retries:
                if attempt == max_retries:
                    print(f"[HONCHO DEBUG] Exhausted all {max_retries + 1} retry attempts")
                # Return appropriate default based on question category
                if category == 5:
                    return "Empty response"
                else:
                    return "No answer found"
            
            # Calculate delay for retry (exponential backoff with jitter)
            if suggested_wait_time:
                delay = suggested_wait_time + random.uniform(0.5, 2.0)
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            
            print(f"[HONCHO DEBUG] Retrying in {delay:.1f} seconds (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)
    
    # This line should be unreachable, but is included to satisfy static type checkers.
    # If control gets here, treat it as an empty response.
    return "Empty response" if category == 5 else "No answer found"


def get_honcho_answers(in_data, out_data, prediction_key, args):
    """Get answers from Honcho for the question-answering task."""
    print(f"[HONCHO DEBUG] Starting Honcho answer generation...")
    start_time = time.time()
    
    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))
    print(f"[HONCHO DEBUG] Processing {len(in_data['qa'])} questions")
    
    # Initialize Honcho client
    print(f"[HONCHO DEBUG] Initializing Honcho client...")
    client_start = time.time()
    honcho = Honcho(
        base_url=HONCHO_BASE_URL,
        environment=HONCHO_ENVIRONMENT,
        timeout=300
    )
    client_time = time.time() - client_start
    print(f"[HONCHO DEBUG] Honcho client initialized in {client_time:.2f} seconds")
    
    # Load mappings
    mappings = load_honcho_mappings()
    mapping_dict = {m['sample_id']: m for m in mappings}
    
    # Get mapping for this sample
    sample_id = in_data['sample_id']
    print(f"[HONCHO DEBUG] Looking up mapping for sample: {sample_id}")
    if sample_id not in mapping_dict:
        raise ValueError(f"Sample {sample_id} not found in Honcho mappings")
    
    mapping = mapping_dict[sample_id]
    app_id = mapping['app_id']
    speakers = mapping['users']
    print(f"[HONCHO DEBUG] Found mapping - App ID: {app_id}, Speakers: {list(speakers.keys())}")
    
    # Prepare list of question indices that still need predictions.
    questions_to_process: list[int] = [
        i for i, qa in enumerate(in_data['qa'])
        if (prediction_key not in out_data['qa'][i]) or args.overwrite
    ]
    print(f"[HONCHO DEBUG] Processing {len(questions_to_process)} questions (after filtering existing predictions)")
    
    # Process questions in batches
    total_questions = len(in_data['qa'])
    questions_processed = 0
    
    # Only iterate over the limited set prepared above
    for batch_start_idx in tqdm(range(0, len(questions_to_process), args.batch_size), desc='Generating answers'):
        batch_start = time.time()
        print(f"[HONCHO DEBUG] Starting batch at index {batch_start_idx}")
        
        questions_batch = []
        include_idxs = []
        
        # Collect questions for this batch
        for i in range(batch_start_idx, min(batch_start_idx + args.batch_size, len(questions_to_process))):
            orig_idx = questions_to_process[i]
            qa = in_data['qa'][orig_idx]
            
            # Skip if already processed and not overwriting
            if prediction_key in out_data['qa'][orig_idx] and not args.overwrite:
                print(f"[HONCHO DEBUG] Skipping question {orig_idx + 1} (already processed)")
                continue
            
            include_idxs.append(orig_idx)
            questions_batch.append(qa)
        
        if not questions_batch:
            print(f"[HONCHO DEBUG] No questions to process in this batch")
            continue
        
        print(f"[HONCHO DEBUG] Processing {len(questions_batch)} questions in this batch (max 10)")
        
        # Process each question
        for idx, qa in zip(include_idxs, questions_batch):
            question_start = time.time()
            question = qa['question']
            category = qa.get('category', 1)
            
            print(f"[HONCHO DEBUG] Question {idx + 1}/{total_questions}: {question}")
            
            # Handle special formatting for temporal questions
            if category == 2:
                question += ' Use DATE of CONVERSATION to answer with an approximate date.'
                print(f"[HONCHO DEBUG] Added temporal context to category 2 question")
            elif category == 5:
                # Build multiple-choice prompt exactly like claude_utils
                base_q = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                if random.random() < 0.5:
                    question = base_q.format('Not mentioned in the conversation', qa.get('answer', qa.get('adversarial_answer', '')))
                    answer_key = {'a': 'Not mentioned in the conversation', 'b': qa.get('answer', qa.get('adversarial_answer', ''))}
                else:
                    question = base_q.format(qa.get('answer', qa.get('adversarial_answer', '')), 'Not mentioned in the conversation')
                    answer_key = {'b': 'Not mentioned in the conversation', 'a': qa.get('answer', qa.get('adversarial_answer', ''))}
                print("[HONCHO DEBUG] Formatted category 5 question with two-choice options")
            else:
                answer_key = None
            
            # Get answer from Honcho
            raw_answer = process_single_question(honcho, app_id, speakers, question, category)
            
            # Map letter to text for category-5
            if category == 5 and answer_key is not None:
                answer = get_cat_5_answer(raw_answer, answer_key)
            else:
                answer = raw_answer.strip()
            
            question_time = time.time() - question_start
            print(f"[HONCHO DEBUG] Question {idx + 1} completed in {question_time:.2f} seconds")
            print(f"[HONCHO DEBUG] Final answer: {answer}")
            
            # Store the answer
            out_data['qa'][idx][prediction_key] = answer.strip()
            questions_processed += 1
            
            # Optionally track context (sessions used)
            if args.use_rag:
                # For Honcho, we could track which sessions were accessed
                # This would require modifying the dialectic API response
                out_data['qa'][idx][prediction_key + '_context'] = []
            # DEBUG: print stored answer summary
            print(f"[HONCHO DEBUG] STORED → idx {idx} | cat {category} | answer '{answer.strip()}'")
        
        batch_time = time.time() - batch_start
        print(f"[HONCHO DEBUG] Batch completed in {batch_time:.2f} seconds")
        
        # Save progress after each batch
        if questions_processed > 0:
            progress_file = getattr(args, 'out_file', 'results/honcho_progress.json')
            try:
                # Create the correct format expected by the main script
                progress_data = [{'sample_id': sample_id, 'qa': out_data['qa']}]
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                print(f"[HONCHO DEBUG] Progress saved to {progress_file}")
            except Exception as e:
                print(f"[HONCHO DEBUG] Warning: Could not save progress: {e}")
    
    total_time = time.time() - start_time
    print(f"[HONCHO DEBUG] Honcho answer generation completed in {total_time:.2f} seconds")
    print(f"[HONCHO DEBUG] Processed {questions_processed} questions total")
    
    # CRITICAL FIX: Ensure ALL questions have 'answer' field as expected by evaluation.py
    # For adversarial questions (category 5), copy 'adversarial_answer' to 'answer'
    questions_fixed = 0
    for qa in out_data['qa']:
        if qa.get('category') == 5 and 'adversarial_answer' in qa and 'answer' not in qa:
            qa['answer'] = qa['adversarial_answer']
            questions_fixed += 1
    print(f"[HONCHO DEBUG] Fixed {questions_fixed} adversarial questions to have 'answer' field")
    
    # Verify all questions now have answer fields
    missing_answer = [i for i, qa in enumerate(out_data['qa']) if 'answer' not in qa]
    if missing_answer:
        print(f"[HONCHO DEBUG] WARNING: {len(missing_answer)} questions still missing 'answer' field: {missing_answer[:10]}")
    else:
        print(f"[HONCHO DEBUG] SUCCESS: All {len(out_data['qa'])} questions now have 'answer' field")
    
    return out_data 