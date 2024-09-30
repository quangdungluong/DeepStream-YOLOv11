#!/bin/bash
DIR=$(pwd)
echo $DIR

echo "Clean nvdsinfer_customparser_yolo bindings"
cd libs && CUDA_VER=$CUDA_VER make -C nvdsinfer_customparser_yolo clean && cd $DIR

echo "Clean nvdsinfer_customparser_yolo_seg bindings"
cd libs && CUDA_VER=$CUDA_VER make -C nvdsinfer_customparser_yolo_seg clean && cd $DIR
