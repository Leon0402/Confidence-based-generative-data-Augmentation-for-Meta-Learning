#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 script_name [arguments...]"
    exit 1
fi

SCRIPT_NAME=$1
# Remove the script name from the arguments list
shift

# Check if the script exists in the temp directory
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "The script $SCRIPT_NAME does not exist."
    exit 1
fi

# Create a temporary directory
TMP_DIR=$(mktemp -d ~/tmp_XXXXXX)
if [ ! -e "$TMP_DIR" ]; then
    echo "Failed to create a temporary directory."
    exit 1
fi

rsync -av --exclude='.git/' ~/Confidence-based-generative-data-Augmentation-for-Meta-Learning/* "$TMP_DIR"
cd "$TMP_DIR"

# Execute the specified script with its arguments
./"$SCRIPT_NAME" "$@"

# Capture the exit status of the script and cleanup
STATUS=$?
cd -
rm -rf "$TMP_DIR"
exit $STATUS
