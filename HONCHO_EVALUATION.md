# Honcho Evaluation for LoCoMo

This implementation enables running the LoCoMo evaluation benchmark on Honcho, a platform for building AI agents with long-term memory and theory-of-mind capabilities.

## Overview

The implementation consists of three main components:

1. **Data Ingestion** - Parse LoCoMo dataset and populate Honcho
2. **Evaluation Module** - Interface with Honcho's dialectic API for question answering
3. **Evaluation Script** - Run the evaluation and generate metrics

## Architecture

### Data Model Mapping

- **LoCoMo samples** (10 conversations) → **Honcho Apps**
- **LoCoMo speakers** (2 per conversation) → **Honcho Users**
- **LoCoMo sessions** → **Honcho Sessions** (preserving temporal boundaries)
- **LoCoMo dialogs** → **Honcho Messages**

### Key Design Decisions

1. **Multiple Sessions**: Each LoCoMo session is preserved as a separate Honcho session to maintain temporal boundaries
2. **Cross-Session Retrieval**: Leverages Honcho's built-in dialectic API which automatically retrieves information across all sessions
3. **Dual Perspectives**: Messages are stored twice - once from each speaker's perspective to enable questions about either participant

## Setup

### Prerequisites

1. Install Honcho Python SDK:
```bash
pip install honcho-ai
```

2. Set up Honcho instance (local or cloud):
   - For local: Follow Honcho's local setup guide
   - For cloud: Sign up at https://honcho.dev

3. Set environment variables:
```bash
export HONCHO_API_KEY="your-api-key"  # or "demo" for local
export HONCHO_ENVIRONMENT="local"      # or "production"
```

### Data Ingestion

Run the ingestion script to populate Honcho with LoCoMo data:

```bash
python scripts/ingest_locomo_to_honcho.py
```

This will:
- Create 10 apps (one per conversation)
- Create 2 users per app (one per speaker)
- Create sessions with proper timestamps
- Store messages with metadata (dialog IDs, dates, etc.)
- Save mappings to `data/honcho_mappings.json`

## Running Evaluation

1. Ensure environment variables are set (see `scripts/env.sh`)

2. Run the evaluation:
```bash
bash scripts/evaluate_honcho.sh
```

This will:
- Load the LoCoMo questions
- Query Honcho for each question
- Generate answers using the dialectic API
- Calculate F1 scores and other metrics
- Save results and statistics

## Implementation Details

### Message Storage

Each message is stored twice to maintain both perspectives:
- Speaker A's message: stored as "user" message for A, "assistant" message for B
- Speaker B's message: stored as "user" message for B, "assistant" message for A

### Question Processing

1. **User Identification**: Determine which speaker the question is about
2. **Session Selection**: Choose appropriate session (dialectic API searches across all)
3. **Query Formation**: Format question with appropriate prompt
4. **Answer Extraction**: Parse response from dialectic API

### Evaluation Metrics

- **F1 Score**: Primary metric comparing predicted vs ground truth answers
- **Recall** (optional): Tracks which sessions were used to answer questions

## Files Created

- `scripts/ingest_locomo_to_honcho.py` - Data ingestion script
- `task_eval/honcho_utils.py` - Honcho evaluation utilities
- `scripts/evaluate_honcho.sh` - Evaluation runner script
- Modified `task_eval/evaluate_qa.py` - Added Honcho support

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure `HONCHO_API_KEY` is set correctly
2. **Connection Errors**: Check if Honcho instance is running (for local)
3. **Missing Mappings**: Run ingestion script before evaluation
4. **Memory Issues**: Process smaller batches by adjusting `--batch-size`

### Debugging

Enable verbose logging:
```bash
export HONCHO_LOG_LEVEL=debug
```

## Future Improvements

1. **Batch Processing**: Implement batch queries to dialectic API for efficiency
2. **Evidence Tracking**: Enhance to track which sessions/messages were used
3. **Context Optimization**: Fine-tune prompts for better answer extraction
4. **Performance Metrics**: Add timing and resource usage tracking 