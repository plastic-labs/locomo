#!/bin/bash
# sets necessary environment variables
source scripts/env.sh

export HONCHO_BASE_URL=http://localhost:8000
export HONCHO_ENVIRONMENT=local
export DATA_FILE_PATH=data/conv-26_only.json
# export DATA_FILE_PATH=data/locomo10.json
export OUT_DIR=results/honcho/baseline_dates_no_explanation

# Evaluate Honcho
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model honcho --batch-size 1