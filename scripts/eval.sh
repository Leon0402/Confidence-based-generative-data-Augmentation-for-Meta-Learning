#!/bin/bash
MODEL_DIR=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_dir) MODEL_DIR="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo "No model directory provided. Use --model_dir to specify the directory."
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_DIR")


echo "Running model $MODEL_DIR in within-domain mode"

for TRAINING_OUTPUT_DIR in "./training_output/$MODEL_NAME/within-domain"/*
do
    echo "Running model with trainin output: $TRAINING_OUTPUT_DIR"

    python -m cdmetadl.eval \
        --model_dir="$MODEL_DIR" \
        --training_output_dir="$TRAINING_OUTPUT_DIR" \
        --test_tasks_per_dataset=100 \
        --verbose 
done

echo "Running model $MODEL_DIR in cross-domain mode" 

for TRAINING_OUTPUT_DIR in "./training_output/$MODEL_NAME/cross-domain"/*
do
    echo "Running model with trainin output: $TRAINING_OUTPUT_DIR"

    python -m cdmetadl.eval \
        --model_dir="$MODEL_DIR" \
        --training_output_dir="$TRAINING_OUTPUT_DIR" \
        --test_tasks_per_dataset=100 \
        --verbose 
done