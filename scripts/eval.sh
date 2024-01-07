#!/bin/bash
MODEL_NAME=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_name) MODEL_NAME="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

if [ -z "$MODEL_NAME" ]; then
    echo "No model directory provided. Use --model_name to specify the directory."
    exit 1
fi

# Delete previous output
rm -r "./output/full/eval"

echo "Running model $MODEL_NAME in within-domain mode"

for TRAINING_OUTPUT_DIR in "./output/full/training/$MODEL_NAME/within-domain"/*
do
    echo "Running model with trainin output: $TRAINING_OUTPUT_DIR"

    python -m cdmetadl.eval \
        --output_dir="output/full/eval" \
        --training_output_dir="$TRAINING_OUTPUT_DIR" \
        --test_tasks_per_dataset=1000 \
        --verbose 
done

echo "Running model $MODEL_NAME in cross-domain mode" 

TRAINING_OUTPUT_DIR="./output/full/training/$MODEL_NAME/cross-domain"
echo "Running model with training output: $TRAINING_OUTPUT_DIR"

python -m cdmetadl.eval \
    --output_dir="output/full/eval" \
    --training_output_dir="$TRAINING_OUTPUT_DIR" \
    --test_tasks_per_dataset=1000 \
    --verbose 