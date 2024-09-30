# DeepStream-YOLOv11

## Introduction
This repository offers plug-and-play custom parsers tailored for YOLOv11 AI models in DeepStream. Ideal for developers looking to streamline model parsing in DeepStream applications.

## Requirements
This repository supports DeepStreamSDK version 6.2, 6.3, 6.4, and 7.0 on Jetson and dGPU platform. You can follow [this guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) for detailed installation instructions.
>Note: I prefer using Docker containers on both dGPU and Jetson platforms. These containers provide a convenient, out-of-the-box way to deploy DeepStream applications by packaging all associated dependencies within the container.

## Supported Models

### Object Detection

- YOLOv11

## Usage

### 1. Clone the repository
```bash
https://github.com/quangdungluong/DeepStream-YOLOv11
cd DeepStream-YOLOv11
```

### 2. Build Custom Parser Function
```bash
bash scripts/compile_nvdsinfer.sh
```

### 3. Convert Models to ONNX
Export the ONNX model file. Check the [documentation](./docs/) for more detailed instructions.

### 4. Convert Models ONNX to TensorRT
Convert ONNX model to a TensorRT engine using [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec).

### 5. Edit the `config_primary_yolov11` file
Edit the `configs/config_primary_yolov11.txt` file according to your model.
```
[property]
...
model-engine-file=yolo11s_b1_fp32.engine
...
num-detected-classes=80
...
# 0: FP32, 1: INT8, 2:FP16
network-mode=0
...
parse-bbox-func-name=NvDsInferParseYolo
...
custom-lib-path=../libs/nvdsinfer_customparser_yolo/libnvds_infercustomparser_yolo.so
```

### 6. Edit the `deepstream_app_config` file
Edit the `deepstream_app_config.txt` file according to your GIE config file.
```
[primary-gie]
...
config-file=config_primary_yolov11.txt
```

### 7. Run DeepStream Application
```bash
deepstream-app -c deepstream_app_config.txt
```

## Reference

- [DeepStream-YOLO](https://github.com/marcoslucianops/DeepStream-Yolo)
