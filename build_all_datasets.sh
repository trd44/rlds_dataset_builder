#!/bin/bash
# Script to build all RLDS datasets
# Usage: bash rlds_dataset_builder/build_all_datasets.sh

# Navigate to rlds_dataset_builder directory
cd "$(dirname "$0")" || exit 1

# ============================================
# CONFIGURATION
# ============================================

# List of dataset folders to build
DATASETS=(
    # "assembly_line_sorting"
    "cube_sorting"
    "height_stacking"
    "pattern_replication"
    "hanoi_50"
    "hanoi4x3_50"
    # "hanoi_300"
    # "hanoi_dataset"
    # "hanoi_fixed_dataset"
    # "example_dataset"  # Uncomment if you want to build this too
)

# ============================================

echo "Building RLDS datasets..."
echo "=========================================="

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Building dataset: $DATASET"
    echo "----------------------------------------"
    
    # Check if directory exists
    if [ ! -d "$DATASET" ]; then
        echo "⚠ Directory $DATASET not found, skipping..."
        continue
    fi
    
    # Navigate into the dataset directory
    cd "$DATASET" || {
        echo "✗ Failed to enter directory $DATASET"
        continue
    }
    
    # Run tfds build
    tfds build --overwrite
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "✓ Successfully built $DATASET"
    else
        echo "✗ Failed to build $DATASET"
    fi
    
    # Navigate back to rlds_dataset_builder directory
    cd ..
    
    echo "=========================================="
done

echo ""
echo "All datasets processed!"

