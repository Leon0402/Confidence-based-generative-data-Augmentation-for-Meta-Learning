#!/bin/bash
DATASETS_DIR=""
TRAINED_MODEL_PATH=""
MODEL=""
CONFIG_PATH=""
OUTPUT_DIR=""
LOG_FILE=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2;;
        --model_path) TRAINED_MODEL_PATH="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --config_path) CONFIG_PATH="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --log_file) LOG_FILE="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

# Check if arguments were provided
if [ -z "$DATASETS_DIR" ]; then
    echo "No datasets directory provided. Use --datasets_dir to specify the directory."
    exit 1
fi

if [ -z "$TRAINED_MODEL_PATH" ]; then
    echo "No trained model directory provided. Use --model_path to specify the directory."
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "No  model name provided. Use --model to specify the directory."
    exit 1
fi

if [ -z "$CONFIG_PATH" ]; then
    echo "No config path provided. Use --config_path to specify the directory."
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "No output dir provided. Use --output_dir to specify the directory."
    exit 1
fi

if [ -z "$LOG_FILE" ]; then
    echo "No log filename provided. Use --log_file to specify the name."
    exit 1
fi

touch $LOG_FILE

for K in "1" "5" "10" "20"
do
    K_CONFIG_PATH=$CONFIG_PATH"_k_"$K".yml"
    K_MODEL_PATH=$TRAINED_MODEL_PATH"_k_"$K"/"$MODEL

    echo "Running model $K_MODEL_PATH for K = $K" >> $LOG_FILE
    echo "Running model $K_MODEL_PATH for K = $K" >> $LOG_FILE

    start=$SECONDS
    ./scripts/eval/eval.sh --datasets_dir $DATASETS_DIR --model_path $K_MODEL_PATH --config_path $K_CONFIG_PATH --output_dir $OUTPUT_DIR --log_file $LOG_FILE
    duration=$(( SECONDS - start ))
    echo "Finished running model $K_MODEL_PATH for K = $K in $duration seconds" >> $LOG_FILE
done
