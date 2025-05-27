#!/usr/bin/env python3
"""
Ingest LoCoMo dataset into Honcho.

This script parses the LoCoMo dataset and creates the appropriate
structure in Honcho:
- Each conversation becomes an app
- Each speaker becomes a user
- Each LoCoMo session becomes a Honcho session
- Messages are duplicated to maintain both speaker perspectives
"""

import json
from datetime import datetime
from typing import Dict, List, Any
import os
import sys
from pathlib import Path

# Add parent directory to path to import Honcho
sys.path.insert(0, str(Path(__file__).parent.parent))

from honcho import Honcho

# Configuration
HONCHO_API_KEY = os.environ.get("HONCHO_API_KEY", "demo")
HONCHO_ENVIRONMENT = os.environ.get("HONCHO_ENVIRONMENT", "local")
LOCOMO_DATA_FILE = "data/locomo10.json"


def parse_datetime(date_str: str) -> str:
    """Convert LoCoMo date format to ISO format."""
    # LoCoMo format: "7 May 2023"
    try:
        dt = datetime.strptime(date_str, "%d %B %Y")
        return dt.isoformat()
    except:
        # If parsing fails, return as is
        return date_str


def extract_speaker_name(dialog: Dict[str, Any]) -> str:
    """Extract the speaker name from a dialog entry."""
    return dialog.get("speaker", "Unknown")


def ingest_conversation(honcho: Honcho, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a single conversation into Honcho."""
    sample_id = sample["sample_id"]
    print(f"\nProcessing {sample_id}...")
    
    # Create app for this conversation
    app = honcho.apps.create(
        name=f"locomo_{sample_id}",
        metadata={
            "sample_id": sample_id,
            "source": "locomo",
            "qa_count": len(sample.get("qa", []))
        }
    )
    print(f"  Created app: {app.name} (ID: {app.id})")
    
    # Get speaker names from first session
    conversation = sample["conversation"]
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    
    # Create users for both speakers
    user_a = honcho.apps.users.create(
        app_id=app.id,
        name=speaker_a,
        metadata={"role": "speaker_a", "sample_id": sample_id}
    )
    user_b = honcho.apps.users.create(
        app_id=app.id,
        name=speaker_b,
        metadata={"role": "speaker_b", "sample_id": sample_id}
    )
    print(f"  Created users: {speaker_a} (ID: {user_a.id}), {speaker_b} (ID: {user_b.id})")
    
    # Process sessions
    session_nums = sorted([
        int(key.split("_")[-1]) 
        for key in conversation.keys() 
        if key.startswith("session_") and not key.endswith("_date_time")
    ])
    
    # Create sessions for both users
    for session_num in session_nums:
        session_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        
        if session_key not in conversation:
            continue
            
        session_data = conversation[session_key]
        session_date = conversation.get(date_key, f"Session {session_num}")
        
        # Create session for user A
        session_a = honcho.apps.users.sessions.create(
            app_id=app.id,
            user_id=user_a.id,
            location_id=f"session_{session_num}",
            metadata={
                "original_session": session_num,
                "date": session_date,
                "sample_id": sample_id
            }
        )
        
        # Create session for user B
        session_b = honcho.apps.users.sessions.create(
            app_id=app.id,
            user_id=user_b.id,
            location_id=f"session_{session_num}",
            metadata={
                "original_session": session_num,
                "date": session_date,
                "sample_id": sample_id
            }
        )
        
        print(f"  Processing session {session_num} ({session_date})...")
        
        # Process messages in this session
        for dialog in session_data:
            speaker = dialog["speaker"]
            text = dialog["text"]
            dia_id = dialog.get("dia_id", "")
            
            # Additional metadata
            metadata = {
                "dia_id": dia_id,
                "original_session": session_num,
                "date": session_date,
                "original_speaker": speaker
            }
            
            # Add image info if present
            if "img_url" in dialog:
                metadata["img_url"] = dialog["img_url"]
            if "blip_caption" in dialog:
                metadata["blip_caption"] = dialog["blip_caption"]
            
            # Store from User A's perspective
            if speaker == speaker_a:
                # A's message as user
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_a.id,
                    session_id=session_a.id,
                    content=text,
                    is_user=True,
                    metadata=metadata
                )
                # A's message as assistant in B's session
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_b.id,
                    session_id=session_b.id,
                    content=text,
                    is_user=False,
                    metadata=metadata
                )
            else:  # speaker == speaker_b
                # B's message as user
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_b.id,
                    session_id=session_b.id,
                    content=text,
                    is_user=True,
                    metadata=metadata
                )
                # B's message as assistant in A's session
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_a.id,
                    session_id=session_a.id,
                    content=text,
                    is_user=False,
                    metadata=metadata
                )
        
        print(f"    Processed {len(session_data)} messages")
    
    print(f"  Completed {sample_id}: {len(session_nums)} sessions")
    
    # Store mapping for evaluation
    return {
        "sample_id": sample_id,
        "app_id": app.id,
        "users": {
            speaker_a: user_a.id,
            speaker_b: user_b.id
        }
    }


def main():
    """Main ingestion function."""
    print(f"Connecting to Honcho (environment: {HONCHO_ENVIRONMENT})...")
    honcho = Honcho(
        api_key=HONCHO_API_KEY,
        environment=HONCHO_ENVIRONMENT
    )
    
    print(f"Loading LoCoMo dataset from {LOCOMO_DATA_FILE}...")
    with open(LOCOMO_DATA_FILE, 'r') as f:
        samples = json.load(f)
    
    print(f"Found {len(samples)} conversations to ingest")
    
    # Process each conversation
    mappings = []
    for sample in samples:
        try:
            mapping = ingest_conversation(honcho, sample)
            mappings.append(mapping)
        except Exception as e:
            print(f"Error processing {sample['sample_id']}: {e}")
            continue
    
    # Save mappings for evaluation
    mapping_file = "data/honcho_mappings.json"
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"\nIngestion complete! Mappings saved to {mapping_file}")
    print(f"Successfully ingested {len(mappings)} out of {len(samples)} conversations")


if __name__ == "__main__":
    main() 