#!/bin/bash
MODEL_PATH=""
CONFIG_PATH=""
OUTPUT_DIR=""
DATA_DIR=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2;;
        --config_path) CONFIG_PATH="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --data_dir) DATA_DIR="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "No model directory provided. Use --model_path to specify the directory."
    exit 1
fi

if [ -z "$CONFIG_PATH" ]; then
    echo "No model directory provided. Use --config_path to specify the directory."
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "No model directory provided. Use --output_dir to specify the directory."
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "No model directory provided. Use --data_dir to specify the directory."
    exit 1
fi

echo "Running model $MODEL_PATH in domain-independent mode" 

TRAINING_OUTPUT_DIR="$MODEL_PATH/domain-independent"
echo "Running model with training output: $TRAINING_OUTPUT_DIR"

python -m cdmetadl.eval \
    --training_output_dir="$TRAINING_OUTPUT_DIR" \
    --data_dir="$DATA_DIR" \
    --config_path="$CONFIG_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --verbose


echo "Running model $MODEL_PATH in cross-domain mode" 

TRAINING_OUTPUT_DIR="$MODEL_PATH/cross-domain"
echo "Running model with training output: $TRAINING_OUTPUT_DIR"

python -m cdmetadl.eval \
    --training_output_dir="$TRAINING_OUTPUT_DIR" \
    --data_dir="$DATA_DIR" \
    --config_path="$CONFIG_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --verbose


echo "Running model $MODEL_PATH in within-domain mode"

for TRAINING_OUTPUT_DIR in "$MODEL_PATH/within-domain"/*
do
    echo "Running model with trainin output: $TRAINING_OUTPUT_DIR"

    python -m cdmetadl.eval \
        --training_output_dir="$TRAINING_OUTPUT_DIR" \
        --data_dir="$DATA_DIR" \
        --config_path="$CONFIG_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --verbose
done