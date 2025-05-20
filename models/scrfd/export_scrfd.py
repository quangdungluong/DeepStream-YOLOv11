import argparse
import os

import torch
from mmdet.core.export.pytorch2onnx import generate_inputs_and_wrap_model


def pytorch2onnx(
    config_path,
    checkpoint_path,
    input_img,
    input_shape,
    opset_version=11,
    output_file="tmp.onnx",
    dynamic=True,
    normalize_cfg=None,
):

    input_config = {
        "input_shape": input_shape,
        "input_path": input_img,
        "normalize_cfg": normalize_cfg,
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tmp_ckpt_file = None

    # remove optimizer for smaller file size
    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]
        tmp_ckpt_file = checkpoint_path + "_slim.pth"
        torch.save(checkpoint, tmp_ckpt_file)
        print("remove optimizer params and save to", tmp_ckpt_file)
        checkpoint_path = tmp_ckpt_file

    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config
    )

    if tmp_ckpt_file is not None:
        os.remove(tmp_ckpt_file)

    # Define input and outputs names, which are required to properly define
    # dynamic axes
    input_names = ["input.1"]
    output_names = [
        "score_8",
        "score_16",
        "score_32",
        "bbox_8",
        "bbox_16",
        "bbox_32",
    ]

    # If model graph contains keypoints strides add keypoints to outputs
    if "stride_kps" in str(model):
        output_names += ["kps_8", "kps_16", "kps_32"]

    # Define dynamic axes for export
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {out: {0: "batch"} for out in output_names}
        dynamic_axes[input_names[0]] = {0: "batch"}

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        verbose=False,
        do_constant_folding=True,
    )
    print(f"Successfully exported ONNX model: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MMDetection models to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--input-img", type=str, help="Images for input")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[-1, -1],
        help="input image size",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[127.5, 127.5, 127.5],
        help="mean value used for preprocess input data",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[128.0, 128.0, 128.0],
        help="variance value used for preprocess input data",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Export SCRFD to ONNX")
    args = parse_args()

    assert args.opset_version == 11, "MMDet only support opset 11 now"

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    assert len(args.mean) == 3
    assert len(args.std) == 3

    dynamic = True

    if input_shape[2] <= 0 or input_shape[3] <= 0:
        input_shape = (1, 3, 640, 640)
        dynamic = True

    normalize_cfg = {"mean": args.mean, "std": args.std}

    if len(args.output_file) == 0:
        cfg_name = args.config.split("/")[-1]
        pos = cfg_name.rfind(".")
        cfg_name = cfg_name[:pos]
        if dynamic:
            args.output_file = "%s_shapeNx%dx%d.onnx" % (
                cfg_name,
                input_shape[2],
                input_shape[3],
            )
        else:
            args.output_file = "%s_shape1x%dx%d.onnx" % (
                cfg_name,
                input_shape[2],
                input_shape[3],
            )

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        output_file=args.output_file,
        dynamic=dynamic,
        normalize_cfg=normalize_cfg,
    )
