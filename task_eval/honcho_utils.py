import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import re
from tqdm import tqdm
from honcho import Honcho
from typing import Dict, List, Any

# Initialize Honcho client
HONCHO_API_KEY = os.environ.get("HONCHO_API_KEY", "demo")
HONCHO_ENVIRONMENT = os.environ.get("HONCHO_ENVIRONMENT", "local")

# Load mappings
MAPPINGS_FILE = "data/honcho_mappings.json"

# Question answering prompts
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

"""


def load_honcho_mappings():
    """Load the mappings created during ingestion."""
    if not os.path.exists(MAPPINGS_FILE):
        raise FileNotFoundError(f"Mappings file not found: {MAPPINGS_FILE}. Run ingest_locomo_to_honcho.py first.")
    
    with open(MAPPINGS_FILE, 'r') as f:
        return json.load(f)


def identify_target_user(question: str, speakers: Dict[str, str]) -> tuple[str, str]:
    """
    Identify which user the question is about.
    Returns (user_name, user_id)
    """
    question_lower = question.lower()
    
    # Check each speaker name in the question
    for speaker_name, user_id in speakers.items():
        if speaker_name.lower() in question_lower:
            return speaker_name, user_id
    
    # Default to first speaker if no specific speaker mentioned
    # This handles general questions like "What happened on X date?"
    first_speaker = list(speakers.keys())[0]
    return first_speaker, speakers[first_speaker]


def get_session_for_question(honcho: Honcho, app_id: str, user_id: str, question: str) -> str:
    """
    Determine which session to query based on the question.
    For now, we'll use a simple approach and let the dialectic API handle cross-session retrieval.
    """
    # List all sessions for the user
    sessions = list(honcho.apps.users.sessions.list(
        app_id=app_id,
        user_id=user_id,
        is_active=True
    ))
    
    if not sessions:
        raise ValueError(f"No sessions found for user {user_id}")
    
    # For questions with temporal context, we could parse dates
    # For now, return the first session and let dialectic handle cross-session
    # The dialectic API will search across all sessions anyway
    return sessions[0].id


def process_single_question(honcho: Honcho, app_id: str, speakers: Dict[str, str], question: str, category: int) -> str:
    """Process a single question using Honcho's dialectic API."""
    # Identify target user
    speaker_name, user_id = identify_target_user(question, speakers)
    
    # Get a session for the user (dialectic will search across all sessions)
    session_id = get_session_for_question(honcho, app_id, user_id, question)
    
    # Use appropriate prompt based on category
    if category == 5:
        # Adversarial questions
        prompt = QA_PROMPT_CAT_5.format(question)
    else:
        prompt = QA_PROMPT.format(question)
    
    # Call dialectic API
    try:
        response = honcho.apps.users.sessions.chat(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            queries=prompt
        )
        
        # Extract answer from response
        answer = response.content.strip()
        
        # Clean up the answer
        if answer.lower().startswith("based on"):
            # Remove preamble if present
            lines = answer.split('\n')
            answer = lines[-1].strip() if lines else answer
        
        return answer
    
    except Exception as e:
        print(f"Error calling dialectic API: {e}")
        return "Error retrieving answer"


def get_honcho_answers(in_data, out_data, prediction_key, args):
    """Get answers from Honcho for the question-answering task."""
    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))
    
    # Initialize Honcho client
    honcho = Honcho(
        api_key=HONCHO_API_KEY,
        environment=HONCHO_ENVIRONMENT
    )
    
    # Load mappings
    mappings = load_honcho_mappings()
    mapping_dict = {m['sample_id']: m for m in mappings}
    
    # Get mapping for this sample
    sample_id = in_data['sample_id']
    if sample_id not in mapping_dict:
        raise ValueError(f"Sample {sample_id} not found in Honcho mappings")
    
    mapping = mapping_dict[sample_id]
    app_id = mapping['app_id']
    speakers = mapping['users']
    
    # Process questions in batches
    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers'):
        questions_batch = []
        include_idxs = []
        
        # Collect questions for this batch
        for i in range(batch_start_idx, min(batch_start_idx + args.batch_size, len(in_data['qa']))):
            qa = in_data['qa'][i]
            
            # Skip if already processed and not overwriting
            if prediction_key in out_data['qa'][i] and not args.overwrite:
                continue
            
            include_idxs.append(i)
            questions_batch.append(qa)
        
        if not questions_batch:
            continue
        
        # Process each question
        for idx, qa in zip(include_idxs, questions_batch):
            question = qa['question']
            category = qa.get('category', 1)
            
            # Handle special formatting for temporal questions
            if category == 2:
                question += ' Use DATE of CONVERSATION to answer with an approximate date.'
            
            # Get answer from Honcho
            answer = process_single_question(honcho, app_id, speakers, question, category)
            
            # Store the answer
            out_data['qa'][idx][prediction_key] = answer.strip()
            
            # Optionally track context (sessions used)
            if args.use_rag:
                # For Honcho, we could track which sessions were accessed
                # This would require modifying the dialectic API response
                out_data['qa'][idx][prediction_key + '_context'] = []
    
    return out_data 