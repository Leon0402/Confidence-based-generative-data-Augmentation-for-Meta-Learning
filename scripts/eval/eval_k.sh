#!/bin/bash
KS="1 5 10"
FORWARD_ARGS=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets_dir|--output_dir)
            FORWARD_ARGS="$FORWARD_ARGS $1 $2"; shift 2;;
        --run_cross_domain|--run_within_domain)
            FORWARD_ARGS="$FORWARD_ARGS $1"; shift 1;;
        --model_path) 
            TRAINED_MODEL_PATH="$2"; shift 2;;
        --model) 
            MODEL="$2"; shift 2;;
        --config_path) 
            CONFIG_PATH="$2"; shift 2;;
        --ks)
            KS="$2"; shift 2;;
        --log_file) 
            LOG_FILE="$2"; shift 2;;
        *)
            echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

# Check if arguments were provided
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

if [ -z "$LOG_FILE" ]; then
    echo "No log filename provided. Use --log_file to specify the name."
    exit 1
fi

touch $LOG_FILE

IFS=' ' read -r -a K_ARRAY <<< "$KS"
for K in "${K_ARRAY[@]}"
do
    K_CONFIG_PATH=$CONFIG_PATH"_k_"$K".yml"
    K_MODEL_PATH=$TRAINED_MODEL_PATH"_k_"$K"/"$MODEL

    echo "Running model $K_MODEL_PATH for K = $K" >> $LOG_FILE
    echo "Running model $K_MODEL_PATH for K = $K" >> $LOG_FILE

    start=$SECONDS
    ./scripts/eval/eval.sh $FORWARD_ARGS --model_path $K_MODEL_PATH --config_path $K_CONFIG_PATH --log_file $LOG_FILE
    duration=$(( SECONDS - start ))
    echo "Finished running model $K_MODEL_PATH for K = $K in $duration seconds" >> $LOG_FILE
done
