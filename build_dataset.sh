#!/bin/bash
# Script to build a single RLDS dataset
# Usage: bash rlds_dataset_builder/build_dataset.sh <dataset_name>

# Check if dataset name was provided
if [ $# -eq 0 ]; then
    echo "Error: No dataset name provided"
    echo "Usage: bash build_dataset.sh <dataset_name>"
    echo ""
    echo "Available datasets:"
    echo "  - assembly_line_sorting"
    echo "  - cube_sorting"
    echo "  - height_stacking"
    echo "  - pattern_replication"
    echo "  - hanoi_300"
    echo "  - hanoi_50"
    echo "  - hanoi4x3_50"
    echo "  - hanoi_dataset"
    echo "  - hanoi_fixed_dataset"
    exit 1
fi

DATASET=$1

# Navigate to rlds_dataset_builder directory
cd "$(dirname "$0")" || exit 1

echo "Building RLDS dataset: $DATASET"
echo "=========================================="

# Check if directory exists
if [ ! -d "$DATASET" ]; then
    echo "Error: Directory $DATASET not found"
    exit 1
fi

# Navigate into the dataset directory
cd "$DATASET" || {
    echo "Error: Failed to enter directory $DATASET"
    exit 1
}

# Run tfds build
echo "Running: tfds build --overwrite"
tfds build --overwrite

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully built $DATASET"
    exit 0
else
    echo ""
    echo "✗ Failed to build $DATASET"
    exit 1
fi

