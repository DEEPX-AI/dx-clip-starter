import os
import argparse
import torch
import onnx
import onnxsim

import open_clip
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OpenCLIP visual model to ONNX format"
    )
    parser.add_argument(
        "--model", type=str, default="ViT-L-14-quickgelu", help="OpenCLIP model name"
    )
    parser.add_argument(
        "--pretrained", type=str, default="dfn2b", help="Pretrained weights tag"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./onnx", help="Directory to save ONNX model"
    )
    return parser.parse_args()


def print_config(args, output_path):
    print("--- Configuration ---")
    print(f"PyTorch Text Model: {args.model} (pretrained: {args.pretrained})")
    print(f"Output path: {output_path}")
    print("---------------------")


def parse_preprocess(preprocess):
    dx_preprocess = []

    for trf in preprocess.transforms:
        trf_name = trf.__class__.__name__

        if trf_name == "Resize":
            size = trf.size
            if isinstance(size, int):
                dx_preprocess.append({"resize": {"width": size, "height": size}})
            else:
                dx_preprocess.append({"resize": {"width": size[0], "height": size[1]}})
        elif trf_name == "ToTensor":
            dx_preprocess.append({"div": {"x": 255.0}})
        elif trf_name == "Normalize":
            mean = trf.mean
            std = trf.std
            dx_preprocess.append(
                {"normalize": {"mean": mean, "std": std}},
            )
        elif trf_name == "function":
            trf_name = trf.__name__
            if trf_name == "_convert_to_rgb":
                dx_preprocess.append({"convertColor": {"form": "BGR2RGB"}})
            else:
                raise NotImplementedError(trf_name)
        elif trf_name == "function":
            trf_name = trf.__name__
            if trf_name == "_convert_to_rgb":
                dx_preprocess.append({"convertColor": {"form": "BGR2RGB"}})
            else:
                raise NotImplementedError(trf_name)

        elif trf_name == "CenterCrop":
            size = trf.size

            if isinstance(size, int):
                dx_preprocess.append({"centercrop": {"width": size, "height": size}})
            else:
                dx_preprocess.append({"centercrop": {"width": size[0], "height": size[1]}})

        else:
            raise NotImplementedError(trf_name)
    return dx_preprocess


def get_config(img: torch.Tensor, preprocess):
    template = {
        "inputs": {"x": [1, 3, 224, 224]},
        "default_loader": {
            "dataset_path": "calibration_dataset/",
            "file_extensions": ["jpeg", "jpg", "png", "JPEG"],
        },
        "calibration_num": 100,
        "calibration_method": "ema",
        "train_batchsize": 32,
        "num_samples": 100,
    }
    #template["inputs"]["input"] = img.shape
    _preprocess = parse_preprocess(preprocess)
    _preprocess.append({"transpose": {"axis": [2, 0, 1]}})
    _preprocess.append({"expandDim": {"axis": 0}})

    template["default_loader"]["preprocessings"] = _preprocess
    return template

def main():
    args = parse_args()
    output_path = os.path.join(args.output_dir, f"{args.model}-{args.pretrained}.onnx")
    config_path = os.path.join(args.output_dir, f"{args.model}-{args.pretrained}.json")
    transform_path = os.path.join(args.output_dir, f"{args.model}-{args.pretrained}-transform.txt")
    print_config(args, output_path)

    # Load model and tokenizer
    print(f"Loading model '{args.model}' ({args.pretrained}) ...")
    model, _, transform = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()

    image = transform(Image.open("assets/img-encoder-sample-1.png")).unsqueeze(0)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.onnx.export(model.visual, image, output_path, opset_version=17)
    with open(transform_path, "w") as f:
        f.write(str(transform))
    print(f"ONNX model saved to: {output_path}")
    config = get_config(image, transform)
    with open(config_path, "w") as f:
        import json
        json.dump(config, f)
    print(f"Config file saved to: {config_path}")
    
    print("Simplifying ONNX model ...")
    onnx_model = onnxsim.simplify(output_path)[0]
    onnx.save(onnx_model, output_path)
    print(f"Simplified ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
