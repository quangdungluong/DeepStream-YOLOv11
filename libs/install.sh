#!/bin/bash
DIR=$(pwd)
echo $DIR

echo "Building nvdsinfer_customparser_yolo bindings"
cd libs && CUDA_VER=$CUDA_VER make -C nvdsinfer_customparser_yolo && cd $DIR