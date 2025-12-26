"""
CLIP Text Encoder ONNX Test - Simple CLI version
Simple educational code for testing ONNX-based CLIP text encoder
Using OpenCLIP models
"""

import torch
import onnxruntime
import os
import sys
import argparse
import numpy as np
import open_clip
import re


class ONNXModel(torch.nn.Module):
    """Wrapper class to use ONNX model like a PyTorch module"""
    
    def __init__(self, model_path: str):
        super().__init__()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        self.model = onnxruntime.InferenceSession(model_path)
        self.output_names = [x.name for x in self.model.get_outputs()]
    
    def forward(self, x):
        """Pass input to ONNX model and return result"""
        onnx_inputs = self.model.get_inputs()
        inputs_dict = {}
        
        # Check if input is tuple/list or single tensor
        is_multiple = isinstance(x, (tuple, list))
        
        if len(onnx_inputs) > 1 and is_multiple:
            # Multiple inputs: add each input to dictionary
            for i, input_node in enumerate(onnx_inputs):
                if i < len(x):
                    inputs_dict[input_node.name] = x[i].cpu().numpy()
        else:
            # Single input
            inputs_dict[onnx_inputs[0].name] = x.cpu().numpy()
        
        # Run inference
        pred = self.model.run(self.output_names, inputs_dict)
        
        # Process result
        if isinstance(pred, list):
            pred = pred[0] if len(pred) == 1 else np.stack(pred)
        
        return torch.from_numpy(pred) if isinstance(pred, np.ndarray) else torch.Tensor(pred)


class TextEncoder(torch.nn.Module):
    """OpenCLIP text encoder ONNX wrapper"""
    
    def __init__(self, onnx_path: str):
        super().__init__()
        self.text_encoder_onnx = ONNXModel(onnx_path)
    
    def forward(self, text_tokens):
        """Convert text tokens to feature vector"""
        # ONNX model takes tokenized text directly as input
        text_features = self.text_encoder_onnx(text_tokens)
        return text_features


def extract_model_name_from_path(onnx_path: str) -> str:
    """Extract model name from ONNX file path
    
    Example: onnx_text/ViT-L-14-quickgelu-dfn2b-text.onnx -> ViT-L-14-quickgelu
    Pattern: {model}-{pretrained}-text.onnx
    """
    filename = os.path.basename(onnx_path)
    # Extract model name from -{pretrained}-text.onnx pattern
    # ViT-L-14-quickgelu-dfn2b-text.onnx -> ViT-L-14-quickgelu
    # Remove the last -{pretrained}-text.onnx part
    match = re.match(r'(.+?)-[^-]+-text\.onnx', filename)
    if match:
        model_name = match.group(1)
        return model_name
    
    # Return default value (when extraction from filename fails)
    return "ViT-L-14-quickgelu"


def get_context_length(model_path: str) -> int:
    """Automatically detect context_length from ONNX model"""
    session = onnxruntime.InferenceSession(model_path)
    inputs = session.get_inputs()
    
    if len(inputs) > 0 and len(inputs[0].shape) >= 2:
        seq_dim = inputs[0].shape[1]
        if isinstance(seq_dim, int):
            return seq_dim
    
    return 77  # CLIP default value


def main():
    parser = argparse.ArgumentParser(description="OpenCLIP Text Encoder ONNX Test")
    parser.add_argument("--onnx-model", type=str,
                       default="onnx/ViT-L-14-quickgelu-dfn2b-text.onnx",
                       help="Text encoder ONNX model path")
    parser.add_argument("--model-name", type=str, default=None,
                       help="OpenCLIP model name (optional, auto-detected if omitted, e.g., ViT-L-14-quickgelu)")
    parser.add_argument("--texts", type=str, nargs="+",
                       default=["a photo of a cat", "a dog", "a diagram"],
                       help="Texts to encode (space-separated)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Don't normalize features")
    parser.add_argument("--show-similarity", action="store_true",
                       help="Show similarity matrix between texts")
    
    args = parser.parse_args()
    
    # Check model file
    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model file not found: {args.onnx_model}")
        sys.exit(1)
    
    # Extract model name
    if args.model_name is None:
        args.model_name = extract_model_name_from_path(args.onnx_model)
        print(f"Auto-detected model name: {args.model_name}")
    
    # Load model
    print("\n[1/5] Loading model...")
    try:
        text_encoder = TextEncoder(args.onnx_model)
        context_length = get_context_length(args.onnx_model)
        print(f"  ✓ ONNX model loaded (context_length={context_length})")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load tokenizer (auto-download if needed)
    print(f"\n[2/5] Loading tokenizer...")
    try:
        tokenizer = open_clip.get_tokenizer(args.model_name)
        print(f"  ✓ Tokenizer loaded")
    except Exception as e:
        print(f"  ✗ Tokenizer loading failed: {e}")
        print(f"  Please check if model name '{args.model_name}' is correct.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare input
    print(f"\n[3/5] Preparing input...")
    texts = args.texts
    print(f"  Input texts: {texts}")
    
    try:
        # open_clip tokenizer takes a list and returns a tensor
        text_tokens = tokenizer(texts)
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.long()
        else:
            text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        print(f"  ✓ Tokenization completed: shape {text_tokens.shape}")
    except Exception as e:
        print(f"  ✗ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run inference
    print(f"\n[4/5] Running inference...")
    try:
        with torch.no_grad():
            # Process individually to avoid batch processing issues
            # (model may be exported with batch size 1 only)
            batch_size = text_tokens.shape[0]
            if batch_size == 1:
                text_features = text_encoder(text_tokens)
            else:
                # Process batch individually
                text_features_list = []
                for i in range(batch_size):
                    single_token = text_tokens[i:i+1]  # [1, seq_len]
                    single_feature = text_encoder(single_token)
                    text_features_list.append(single_feature)
                text_features = torch.cat(text_features_list, dim=0)
        print(f"  ✓ Inference completed: shape {text_features.shape}")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Normalize
    if not args.no_normalize:
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Display results
    print(f"\n[5/5] Results")
    print("=" * 60)
    
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: \"{text}\"")
        print(f"  Feature shape: {text_features[i].shape}")
        print(f"  Feature norm: {text_features[i].norm().item():.6f}")
        feature_values = text_features[i].cpu().numpy()[:10]
        print(f"  First 10 values: {np.array2string(feature_values, precision=4, suppress_small=True)}")
    
    # 유사도 행렬
    if len(texts) > 1 and args.show_similarity:
        print(f"\nSimilarity Matrix:")
        similarity = text_features @ text_features.t()
        similarity_np = similarity.cpu().numpy()
        
        print("  " + "".join([f"{'Text ' + str(i+1):>10}" for i in range(len(texts))]))
        for i in range(len(texts)):
            row = f"  Text {i+1}: " + " ".join([f"{sim:>9.4f}" for sim in similarity_np[i]])
            print(row)

if __name__ == "__main__":
    main()

