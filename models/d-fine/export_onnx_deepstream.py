import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import argparse
import warnings

import torch
import torch.nn as nn
from src.core import YAMLConfig


def suppress_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x["pred_boxes"]
        scores, classes = torch.max(x["pred_logits"], dim=2, keepdim=True)
        return boxes, scores, classes


def main(args):
    suppress_warnings()
    cfg = YAMLConfig(args.config, resume=args.weights)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)
    else:
        print("not load model.state_dict, use default init state dict...")

    img_size = args.size * 2 if len(args.size) == 1 else args.size
    device = torch.device("cpu")
    onnx_input_img = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = os.path.basename(args.weights).replace(".pth", ".onnx")

    dynamic_axes = {
        "input": {0: "batch"},
        "boxes": {0: "batch"},
        "scores": {0: "batch"},
        "classes": {0: "batch"},
    }

    model = cfg.model.deploy()
    model = nn.Sequential(model, DeepStreamOutput())

    torch.onnx.export(
        model,
        onnx_input_img,
        onnx_output_file,
        input_names=["input"],
        output_names=["boxes", "scores", "classes"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx

        model = onnx.load(onnx_output_file)
        onnx.checker.check_model(model)
        print("ONNX model is valid")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--check", action="store_true", default=False, help="Check onnx model"
    )
    parser.add_argument("-w", "--weights", type=str, required=True, help="Weights file")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=[640],
        help="Inference size [H,W] (default [640])",
    )
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
