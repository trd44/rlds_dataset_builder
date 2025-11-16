#!/usr/bin/env python3
"""
Simple test script to demonstrate how to use the Assembly Line Sorting dataset builder.
"""

# Import and configure TensorFlow first
from tf_config import configure_tensorflow
configure_tensorflow()

import tensorflow as tf
import tensorflow_datasets as tfds
from assembly_line_sorting_dataset_builder import AssemblyLineSorting

def main():
    print("Testing Assembly Line Sorting dataset builder...")
    
    # Create the dataset builder
    builder = AssemblyLineSorting()
    
    # Download and prepare the dataset
    builder.download_and_prepare()
    
    # Get dataset info
    info = builder.info
    print(f"Dataset info: {info}")
    
    # Create a dataset object
    dataset = builder.as_dataset(split='train')
    
    # Take a few examples
    for i, example in enumerate(dataset.take(3)):
        print(f"\nExample {i+1}:")
        print(f"  Steps: {len(example['steps'])}")
        print(f"  First step observation keys: {list(example['steps'][0]['observation'].keys())}")
        print(f"  Language instruction: {example['steps'][0]['language_instruction']}")
        print(f"  File path: {example['episode_metadata']['file_path']}")

if __name__ == "__main__":
    main()
