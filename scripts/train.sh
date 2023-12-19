#!/bin/bash
DATASETS_DIR=""
MODEL_DIR=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2;;
        --model_dir) MODEL_DIR="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

# Check if DATASETS_DIR and MODEL_DIR were provided
if [ -z "$DATASETS_DIR" ]; then
    echo "No datasets directory provided. Use --datasets_dir to specify the directory."
    exit 1
fi

if [ -z "$MODEL_DIR" ]; then
    echo "No model directory provided. Use --model_dir to specify the directory."
    exit 1
fi


echo "Running model $MODEL_DIR in within-domain mode"

for DATASET_PATH in "$DATASETS_DIR"/*
do
    DATASET_NAME=$(basename "$DATASET_PATH")
    echo "Running model with dataset: $DATASET_NAME"

    python -m cdmetadl.train \
        --model_dir="$MODEL_DIR" \
        --domain_type="within-domain" \
        --datasets="$DATASET_NAME" \
        --data_dir="$DATASETS_DIR" \
        --overwrite_previous_results \
        --verbose
done

echo "Running model $MODEL_DIR in cross-domain mode" 

python -m cdmetadl.train \
    --model_dir="$MODEL_DIR" \
    --domain_type="cross-domain" \
    --data_dir="$DATASETS_DIR" \
    --overwrite_previous_results \
    --verbose