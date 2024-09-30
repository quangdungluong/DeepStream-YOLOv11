#!/bin/bash

# Detect installed CUDA version
cuda_dir=$(ls -d /usr/local/cuda-*/ 2>/dev/null | grep -oP 'cuda-\K[0-9]+\.[0-9]+' | tail -n 1)

if [ -z "$cuda_dir" ]; then
    echo "No CUDA installation found in /usr/local/"
    exit 1
else
    echo "Detected CUDA version: $cuda_dir"
fi

# Set the CUDA_VER environment variable
export CUDA_VER=$cuda_dir

# Execute the make command with the detected CUDA version
bash libs/install.sh
