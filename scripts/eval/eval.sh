#!/bin/bash
TRAINED_MODEL_PATH=""
CONFIG_PATH=""
OUTPUT_DIR=""
DATASETS_DIR=""
LOG_FILE=""
RUN_CROSS_DOMAIN=false
RUN_WITHIN_DOMAIN=false

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_path) TRAINED_MODEL_PATH="$2"; shift 2;;
        --config_path) CONFIG_PATH="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --datasets_dir) DATASETS_DIR="$2"; shift 2;;
        --log_file) LOG_FILE="$2"; shift 2;;
        --run_cross_domain) RUN_CROSS_DOMAIN=true; shift;;
        --run_within_domain) RUN_WITHIN_DOMAIN=true; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

if [ -z "$TRAINED_MODEL_PATH" ]; then
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

if [ -z "$DATASETS_DIR" ]; then
    echo "No dataset directory provided. Use --datasets_dir to specify the directory."
    exit 1
fi

if [ -z "$LOG_FILE" ]; then
    echo "No log filename provided. Use --log_file to specify the name."
    exit 1
fi

touch $LOG_FILE

# echo "Running model $TRAINED_MODEL_PATH in domain-independent mode" 
# echo "Running model $TRAINED_MODEL_PATH in domain-independent mode"  >> $LOG_FILE

# start=$SECONDS
# TRAINING_OUTPUT_DIR="$TRAINED_MODEL_PATH/domain-independent"
# python -m cdmetadl.eval \
#     --training_output_dir="$TRAINING_OUTPUT_DIR" \
#     --data_dir="$DATASETS_DIR" \
#     --config_path="$CONFIG_PATH" \
#     --output_dir="$OUTPUT_DIR" \
#     --verbose
# duration=$(( SECONDS - start ))
# echo Task took $duration seconds >> $LOG_FILE

# Cross-domain evaluation
if [ "$RUN_CROSS_DOMAIN" = true ]; then
    echo "Running model $TRAINED_MODEL_PATH in cross-domain mode" 
    echo "Running model $TRAINED_MODEL_PATH in cross-domain mode" >> $LOG_FILE

    start=$SECONDS
    TRAINING_OUTPUT_DIR="$TRAINED_MODEL_PATH/cross-domain"
    python -m cdmetadl.eval \
        --training_output_dir="$TRAINING_OUTPUT_DIR" \
        --data_dir="$DATASETS_DIR" \
        --config_path="$CONFIG_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --verbose
    duration=$(( SECONDS - start ))
    echo "Task took $duration seconds" >> $LOG_FILE
fi

# Within-domain evaluation
if [ "$RUN_WITHIN_DOMAIN" = true ]; then
    echo "Running model $TRAINED_MODEL_PATH in within-domain mode"
    echo "Running model $TRAINED_MODEL_PATH in within-domain mode" >> $LOG_FILE

    for TRAINING_OUTPUT_DIR in "$TRAINED_MODEL_PATH/within-domain"/*
    do
        echo "Running model $TRAINING_OUTPUT_DIR" >> $LOG_FILE
        start=$SECONDS

        python -m cdmetadl.eval \
            --training_output_dir="$TRAINING_OUTPUT_DIR" \
            --data_dir="$DATASETS_DIR" \
            --config_path="$CONFIG_PATH" \
            --output_dir="$OUTPUT_DIR" \
            --verbose
        duration=$(( SECONDS - start ))
        echo "Task took $duration seconds" >> $LOG_FILE
    done
fi