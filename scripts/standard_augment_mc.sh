GPU=""
MODEL=""

# Parse named arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

# Check if arguments were provided
if [ -z "$GPU" ]; then
    echo "No model provided. Use --model to specify model."
    exit 1
fi

if [ -z "$GPU" ]; then
    echo "No gpu provided. Use --gpu to specify gpu."
    exit 1
fi

LOG_FILE=standard_augmentation_mc_$MODEL.log
touch $LOG_FILE

for K in "1" "5" "10" "15" "20"
do
    echo "Running model $MODEL for K = $K"
    echo "Running model $MODEL for K = $K" >> $LOG_FILE
    start=$SECONDS

    CUDA_VISIBLE_DEVICES=$GPU ./scripts/eval.sh --model_path "/fastdata/vilab24/output/full/training/baseline_k_$K/$MODEL" \
    --data_dir "/fastdata/vilab24/meta-album" \
    --config_path "configs/eval/standard_augmentation/mc_dropout/standard_augment_mc_k_$K.yml" \
    --output_dir "/fastdata/vilab24/output/full/eval"
    duration=$(( SECONDS - start ))
    
    echo Task took $duration seconds >> $LOG_FILE
done
