#!/bin/bash

# Inference script for the fusion MLP model

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 --checkpoint CHECKPOINT_PATH --images IMAGE1 [IMAGE2 ...] [--config CONFIG] [--output_dir OUTPUT_DIR]"
    exit 1
fi

# Default values
CONFIG="configs/default.yaml"
OUTPUT_DIR="./outputs"
CHECKPOINT=""
IMAGES=()
SAVE_MESH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --images)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                IMAGES+=("$1")
                shift
            done
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save_mesh)
            SAVE_MESH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ] || [ ${#IMAGES[@]} -eq 0 ]; then
    echo "Error: --checkpoint and --images are required"
    exit 1
fi

echo "Running inference..."
echo "Checkpoint: $CHECKPOINT"
echo "Images: ${IMAGES[*]}"
echo "Output directory: $OUTPUT_DIR"

if [ "$SAVE_MESH" = true ]; then
    python -m src.inference.infer \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --images "${IMAGES[@]}" \
        --output_dir "$OUTPUT_DIR" \
        --save_mesh
else
    python -m src.inference.infer \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --images "${IMAGES[@]}" \
        --output_dir "$OUTPUT_DIR"
fi

echo "Inference completed! Results saved to $OUTPUT_DIR"

