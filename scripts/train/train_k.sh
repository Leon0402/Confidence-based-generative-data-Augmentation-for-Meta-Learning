#!/bin/bash
DATASETS_DIR=""
MODEL_DIR=""
CONFIG_PATH=""
OUTPUT_DIR=""
LOG_FILE=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2;;
        --model_dir) MODEL_DIR="$2"; shift 2;;
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

if [ -z "$MODEL_DIR" ]; then
    echo "No model directory provided. Use --model_dir to specify the directory."
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
    echo "Running model $MODEL_DIR for K = $K" >> $LOG_FILE
    echo "Running model $MODEL_DIR for K = $K" >> $LOG_FILE
    start=$SECONDS

    K_CONFIG_PATH=$CONFIG_PATH"_k_"$K".yml"
    ./scripts/train/train.sh --datasets_dir $DATASETS_DIR --model_dir $MODEL_DIR --config_path $K_CONFIG_PATH --output_dir $OUTPUT_DIR --log_file $LOG_FILE
    duration=$(( SECONDS - start ))
    echo Task took $duration seconds >> $LOG_FILE
done
