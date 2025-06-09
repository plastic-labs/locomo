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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, cast, Literal
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path to import Honcho
sys.path.insert(0, str(Path(__file__).parent.parent))

from honcho import Honcho

# Configuration
HONCHO_BASE_URL = "http://localhost:8000"
HONCHO_ENVIRONMENT = os.environ.get("HONCHO_ENVIRONMENT", "local")
# LOCOMO_DATA_FILE = "data/locomo10.json"
DEFAULT_DATA_FILE = "data/conv-26_only.json"


def parse_session_datetime(date_str: str) -> datetime:
    """Parse strings like '1:56 pm on 8 May, 2023' into a UTC-aware datetime.

    The LoCoMo JSON encodes each session's start time with that exact pattern.
    If parsing fails we fall back to *now* just so the script never crashes.
    """
    try:
        # Normalise AM/PM capitalization just in case ("pm" -> "PM")
        parts = date_str.split()
        if parts[1].lower() in {"am", "pm"}:
            parts[1] = parts[1].upper()
            date_str = " ".join(parts)

        dt = datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        # Fallback – shouldn't happen in clean datasets
        print(f"Failed to parse datetime: {date_str}")
        return datetime.now(tz=timezone.utc)


def extract_speaker_name(dialog: Dict[str, Any]) -> str:
    """Extract the speaker name from a dialog entry."""
    return dialog.get("speaker", "Unknown")


def ingest_conversation(honcho: Honcho, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a single conversation into Honcho."""
    sample_id = sample["sample_id"]
    print(f"\nProcessing {sample_id}...")
    
    # Create app for this conversation
    app = honcho.apps.get_or_create(
        name=f"locomo_{sample_id}",
    )
    print(f"  Created app: {app.name} (ID: {app.id})")
    
    # Get speaker names from first session
    conversation = sample["conversation"]
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    
    # Create users for both speakers
    user_a = honcho.apps.users.get_or_create(
        app_id=app.id,
        name=speaker_a,
        # metadata={"role": "speaker_a", "sample_id": sample_id}
    )
    user_b = honcho.apps.users.get_or_create(
        app_id=app.id,
        name=speaker_b,
        # metadata={"role": "speaker_b", "sample_id": sample_id}
    )
    print(f"  Created users: {speaker_a} (ID: {user_a.id}), {speaker_b} (ID: {user_b.id})")
    
    # Process sessions
    session_nums = sorted(
        int(key.split("_")[-1])
        for key in conversation.keys()
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    
    # Create sessions for both users
    for session_num in session_nums:
        session_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        
        if session_key not in conversation:
            continue
            
        session_data = conversation[session_key]
        raw_session_dt = conversation.get(date_key, "")
        base_dt = parse_session_datetime(raw_session_dt) if raw_session_dt else datetime.now(tz=timezone.utc)
        
        # Create session for user A
        session_a = honcho.apps.users.sessions.create(
            app_id=app.id,
            user_id=user_a.id,
            metadata={
                "original_session": session_num,
                "session_timestamp": base_dt.isoformat(),
                "sample_id": sample_id
            }
        )
        
        # Create session for user B
        session_b = honcho.apps.users.sessions.create(
            app_id=app.id,
            user_id=user_b.id,
            metadata={
                "original_session": session_num,
                "session_timestamp": base_dt.isoformat(),
                "sample_id": sample_id
            }
        )
        
        print(f"  Processing session {session_num} ({base_dt})...")
        
        # Process messages in this session
        for idx, dialog in enumerate(session_data):
            speaker = dialog["speaker"]
            text = dialog["text"]
            dia_id = dialog.get("dia_id", "")
            
            created_at_ts = base_dt + timedelta(minutes=idx)

            metadata = {
                "dia_id": dia_id,
                "original_session": session_num,
                "session_timestamp": created_at_ts.isoformat(),
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
                    metadata=metadata,
                    created_at=created_at_ts
                )
                # A's message as assistant in B's session
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_b.id,
                    session_id=session_b.id,
                    content=text,
                    is_user=False,
                    metadata=metadata,
                    created_at=created_at_ts
                )
            else:  # speaker == speaker_b
                # B's message as user
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_b.id,
                    session_id=session_b.id,
                    content=text,
                    is_user=True,
                    metadata=metadata,
                    created_at=created_at_ts
                )
                # B's message as assistant in A's session
                honcho.apps.users.sessions.messages.create(
                    app_id=app.id,
                    user_id=user_a.id,
                    session_id=session_a.id,
                    content=text,
                    is_user=False,
                    metadata=metadata,
                    created_at=created_at_ts
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
    parser = argparse.ArgumentParser(description="Ingest LoCoMo data into Honcho")
    parser.add_argument("--file", "-f", default=DEFAULT_DATA_FILE, help="Path to LoCoMo JSON file")
    args = parser.parse_args()

    data_file = args.file

    print(f"Connecting to Honcho (environment: {HONCHO_ENVIRONMENT})...")
    honcho = Honcho(
        base_url=HONCHO_BASE_URL,
        environment=cast(Literal["demo", "local", "production"], HONCHO_ENVIRONMENT)
    )
    
    print(f"Loading LoCoMo dataset from {data_file}...")
    with open(data_file, 'r') as f:
        samples = json.load(f)
    
    print(f"Found {len(samples)} conversations to ingest")
    
    # Process each conversation
    mappings = []
    for sample in tqdm(samples):
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