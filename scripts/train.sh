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

echo "Running model $MODEL_DIR in domain-independent mode" 

python -m cdmetadl.train \
    --config_path="configs/full.yml" \
    --model_dir="$MODEL_DIR" \
    --output_dir="./output/full/training" \
    --domain_type="domain-independent" \
    --data_dir="$DATASETS_DIR" \
    --verbose

echo "Running model $MODEL_DIR in cross-domain mode" 

python -m cdmetadl.train \
    --config_path="configs/full.yml" \
    --model_dir="$MODEL_DIR" \
    --output_dir="./output/full/training" \
    --domain_type="cross-domain" \
    --data_dir="$DATASETS_DIR" \
    --verbose

echo "Running model $MODEL_DIR in within-domain mode"

# TODO: Read datasets from config perhaps?
for DATASET_NAME in "AWA" "FNG" "PRT" "BTS" "ACT_410"
do
    echo "Running model with dataset: $DATASET_NAME"

    python -m cdmetadl.train \
        --config_path="configs/full.yml" \
        --model_dir="$MODEL_DIR" \
        --output_dir="./output/full/training" \
        --datasets="$DATASET_NAME" \
        --domain_type="within-domain" \
        --data_dir="$DATASETS_DIR" \
        --verbose 
done