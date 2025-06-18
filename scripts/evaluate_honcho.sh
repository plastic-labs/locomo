#!/bin/bash
# sets necessary environment variables
source scripts/env.sh

# Allow overriding the backend port via $HONCHO_PORT (defaults to 8000)
HONCHO_PORT=${HONCHO_PORT:-8000}
export HONCHO_BASE_URL="http://localhost:${HONCHO_PORT}"
export HONCHO_ENVIRONMENT=local
export DATA_FILE_PATH=data/conv-26_only.json
# export DATA_FILE_PATH=data/locomo10.json
export OUT_DIR=results/${1}
echo "Evaluating Honcho on $DATA_FILE_PATH with output directory $OUT_DIR and backend at $HONCHO_BASE_URL"

# Evaluate Honcho
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model honcho --batch-size 1 --scoring-modes llm f1 --override-cached-scores