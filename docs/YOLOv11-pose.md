# YOLOv11-pose ONNX TensorRT Exporter

### Convert model

#### 1. Install the requirements
```
cd models/yolov11-pose
pip3 install -r requirements.txt
```

#### 2. Download the model
Download the `pt` file from [ultralytics](https://github.com/ultralytics/ultralytics) model zoo.

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
```

#### 3. Export ONNX model
```bash
python3 export_yolov11_pose.py -w yolo11s-pose.pt
```

#### 4. Export TensorRT model
```bash
trtexec --onnx=./yolo11s-pose.onnx \
        --saveEngine=./yolo11s-pose_b1_fp32.engine
```

#### 5. Copy the generated files
Copy the generated ONNX model file, TensorRT model file, and `labels.txt` to the `configs` folder.
