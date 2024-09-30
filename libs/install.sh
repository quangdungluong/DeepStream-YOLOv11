#!/bin/bash
DIR=$(pwd)
echo $DIR

echo "Building nvdsinfer_customparser_yolo bindings"
cd libs && CUDA_VER=$CUDA_VER make -C nvdsinfer_customparser_yolo && cd $DIR

echo "Building nvdsinfer_customparser_yolo_seg bindings"
cd libs && CUDA_VER=$CUDA_VER make -C nvdsinfer_customparser_yolo_seg && cd $DIR

echo "Building nvdsinfer_customparser_yolo_pose bindings"
cd libs && CUDA_VER=$CUDA_VER make -C nvdsinfer_customparser_yolo_pose && cd $DIR
