#!/bin/bash

# Training script for the fusion MLP model

set -e

# Default values
CONFIG="configs/default.yaml"
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting training with config: $CONFIG"
if [ -n "$RESUME" ]; then
    echo "Resuming from checkpoint: $RESUME"
    python -m src.training.train --config "$CONFIG" --resume "$RESUME"
else
    python -m src.training.train --config "$CONFIG"
fi

echo "Training completed!"

