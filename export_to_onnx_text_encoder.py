import os
import argparse
import torch
import onnx
import onnxsim

import open_clip

# 1. Wrapper class to extract only the text encoder
class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        # Call OpenCLIP's text encoding method
        return self.model.encode_text(text)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OpenCLIP Text Model to ONNX format"
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

def main():
    args = parse_args()
    output_path = os.path.join(args.output_dir, f"{args.model}-{args.pretrained}-text.onnx")
    
    print(f"Loading model '{args.model}' ({args.pretrained}) ...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()

    # 2. Load tokenizer and create dummy input
    # Text encoder takes 'tokenized text (integer tensor)' as input, not images
    tokenizer = open_clip.get_tokenizer(args.model)
    dummy_text = tokenizer(["a photo of a cat"]) # Shape: [1, 77] (typically 77 context length)

    # 3. Wrap with wrapper class
    text_encoder = TextEncoderWrapper(model)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Exporting to ONNX ...")
    torch.onnx.export(
        text_encoder,               # Text encoder module
        dummy_text,                 # Dummy text input
        output_path,
        opset_version=18,
        input_names=["text_encoder_input"],       # Input node name (specific)
        output_names=["text_encoder_embedding"], # Output node name (specific, prevent duplicates)
        dynamic_axes={
            "text_encoder_input": {0: "batch_size"},      # Allow variable batch size
            "text_encoder_embedding": {0: "batch_size"}
        }
    )
    print(f"ONNX model saved to: {output_path}")

    # 4. ONNX Simplify
    print("Simplifying ONNX model ...")
    try:
        # Load and validate model first
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ Original model is valid")
        
        # Perform simplification
        onnx_model, check = onnxsim.simplify(onnx_model)
        if check:
            # Validate after simplification
            try:
                onnx.checker.check_model(onnx_model)
                onnx.save(onnx_model, output_path)
                print(f"  ✓ Simplified ONNX model saved to: {output_path}")
            except Exception as e:
                print(f"  ⚠ Simplified model validation failed: {e}")
                print(f"  → Keeping original model (simplification skipped)")
                # Reload and save original model
                onnx_model = onnx.load(output_path)
                onnx.save(onnx_model, output_path)
        else:
            print(f"  ⚠ Model could not be simplified, but original model is valid")
    except Exception as e:
        print(f"Warning: onnxsim failed ({e}), skipping simplification.")
        # Check if original model is valid
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("  ✓ Original model is valid (simplification skipped)")
        except Exception as e2:
            print(f"  ✗ Original model validation failed: {e2}")
            raise

if __name__ == "__main__":
    main()
