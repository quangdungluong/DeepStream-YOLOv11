# D-FINE-detection ONNX TensorRT Exporter

### Convert model

#### 1. Download the D-FINE repository and install the requirements
```bash
cd models/d-fine
git clone https://github.com/Peterande/D-FINE.git
cd D-FINE
pip3 install -r requirements.txt
```

#### 2. Update converter patch

Copy the `export_onnx_deepstream.py` file from `models/d-fine` directory to the `models/d-fine/D-FINE/tools/deployment` directory.

#### 3. Download the model

```bash
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth
```

#### 4. Export ONNX model

```bash
python3 export_onnx_deepstream.py -c configs/define/dfine_hgnetv2_n_coco.yml \
                                -w dfine_n_coco.pth \
                                --check
```

#### 5. Export TensorRT model

```bash
trtexec --onnx=./dfine_n_coco.onnx \
        --saveEngine=./dfine_n_coco_b1_fp32.engine \
        --verbose
```

#### 6. Copy the generated files
Copy the generated ONNX model file, TensorRT model file, and `labels.txt` to the `configs` folder.
