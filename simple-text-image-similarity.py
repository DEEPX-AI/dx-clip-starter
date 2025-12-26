"""
CLIP Text-Image Similarity Calculator
Script to calculate similarity between text and images
"""

import torch
import onnxruntime
import os
import sys
import argparse
import numpy as np
import open_clip
import re
from PIL import Image
from dx_engine import InferenceEngine
from torchvision import transforms


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
        
        if len(onnx_inputs) > 1 and isinstance(x, (tuple, list)):
            for i, input_node in enumerate(onnx_inputs):
                if i < len(x):
                    inputs_dict[input_node.name] = x[i].cpu().numpy()
        else:
            inputs_dict[onnx_inputs[0].name] = x.cpu().numpy()
        
        pred = self.model.run(self.output_names, inputs_dict)
        
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
        text_features = self.text_encoder_onnx(text_tokens)
        return text_features


class ImageEncoder:
    """Image encoder (DXNN-based)"""
    
    def __init__(self, dxnn_path: str):
        if not os.path.isfile(dxnn_path):
            raise FileNotFoundError(f"DXNN model file not found: {dxnn_path}")
        
        self.engine = InferenceEngine(dxnn_path)
        self.input_info = self.engine.get_input_tensors_info()
    
    def encode(self, image_array: np.ndarray):
        """Convert image array to feature vector"""
        outputs = self.engine.run(image_array)
        
        if isinstance(outputs, list):
            image_features = outputs[0]
        else:
            image_features = outputs
        
        return np.array(image_features)


def extract_model_name_from_path(onnx_path: str) -> str:
    """Extract model name from ONNX file path"""
    filename = os.path.basename(onnx_path)
    match = re.match(r'(.+?)-[^-]+-text\.onnx', filename)
    if match:
        return match.group(1)
    return "ViT-L-14-quickgelu"


def create_image_transform():
    """Create CLIP image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, 
                         max_size=None, antialias=True),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_and_preprocess_image(image_path: str, transform):
    """Load and preprocess image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        image_array = image_tensor.numpy()
        image_array = np.expand_dims(image_array, axis=0)  # [1, C, H, W]
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def normalize_features(features):
    """L2 normalization"""
    if isinstance(features, torch.Tensor):
        return features / features.norm(dim=-1, keepdim=True)
    else:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / norms


def compute_similarity_matrix(text_features, image_features):
    """Calculate similarity matrix between text and images"""
    # Text: [num_texts, dim], Image: [dim]
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.cpu().numpy()
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.cpu().numpy()
    
    # Convert image to [1, dim] shape
    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)
    
    # Calculate cosine similarity (already normalized, so just dot product)
    similarity = np.dot(text_features, image_features.T)  # [num_texts, 1]
    return similarity.squeeze()  # [num_texts]


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Text-Image Similarity Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 simple-text-image-similarity.py \\
    --text-encoder onnx_text/ViT-L-14-quickgelu-dfn2b-text.onnx \\
    --image-encoder dxnn/ViT-L-14-quickgelu-dfn2b.dxnn \\
    --texts "a cat" "a dog" "a bird" \\
    --image photo.jpg
        """
    )
    
    parser.add_argument("--text-encoder", type=str,
                       default="onnx/ViT-L-14-quickgelu-dfn2b-text.onnx",
                       help="Text encoder ONNX model path")
    parser.add_argument("--image-encoder", type=str,
                       default="./dxnn/ViT-L-14-quickgelu-dfn2b.dxnn",
                       help="Image encoder DXNN model path")
    parser.add_argument("--model-name", type=str, default=None,
                       help="OpenCLIP model name (optional, auto-detected if omitted)")
    parser.add_argument("--texts", type=str, nargs="+", required=True,
                       help="Texts to encode (space-separated)")
    parser.add_argument("--image", type=str, required=True,
                       help="Image file path to encode")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Don't normalize features")
    
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.text_encoder):
        print(f"Error: Text encoder model file not found: {args.text_encoder}")
        sys.exit(1)
    
    if not os.path.exists(args.image_encoder):
        print(f"Error: Image encoder model file not found: {args.image_encoder}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    print("=" * 70)
    print("CLIP Text-Image Similarity Calculator")
    print("=" * 70)
    
    # Extract model name
    if args.model_name is None:
        args.model_name = extract_model_name_from_path(args.text_encoder)
    
    # [1/5] Load text encoder
    print("\n[1/5] Loading text encoder...")
    try:
        text_encoder = TextEncoder(args.text_encoder)
        print(f"  ✓ Text encoder loaded: {args.text_encoder}")
    except Exception as e:
        print(f"  ✗ Text encoder loading failed: {e}")
        sys.exit(1)
    
    # [2/5] Load image encoder
    print("\n[2/5] Loading image encoder...")
    try:
        image_encoder = ImageEncoder(args.image_encoder)
        print(f"  ✓ Image encoder loaded: {args.image_encoder}")
    except Exception as e:
        print(f"  ✗ Image encoder loading failed: {e}")
        sys.exit(1)
    
    # [3/5] Load tokenizer
    print("\n[3/5] Loading tokenizer...")
    try:
        tokenizer = open_clip.get_tokenizer(args.model_name)
        print(f"  ✓ Tokenizer loaded")
    except Exception as e:
        print(f"  ✗ Tokenizer loading failed: {e}")
        sys.exit(1)
    
    # [4/5] Prepare input and encode
    print("\n[4/5] Preparing input and encoding...")
    
    # Tokenize and encode text
    print(f"  Text input: {args.texts}")
    try:
        text_tokens = tokenizer(args.texts)
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.long()
        else:
            text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        with torch.no_grad():
            batch_size = text_tokens.shape[0]
            if batch_size == 1:
                text_features = text_encoder(text_tokens)
            else:
                text_features_list = []
                for i in range(batch_size):
                    single_token = text_tokens[i:i+1]
                    single_feature = text_encoder(single_token)
                    text_features_list.append(single_feature)
                text_features = torch.cat(text_features_list, dim=0)
        
        if not args.no_normalize:
            text_features = normalize_features(text_features)
        
        print(f"  ✓ Text encoding completed: shape {text_features.shape}")
    except Exception as e:
        print(f"  ✗ Text encoding failed: {e}")
        sys.exit(1)
    
    # Load and encode image
    print(f"  Image input: {args.image}")
    try:
        transform = create_image_transform()
        image_array = load_and_preprocess_image(args.image, transform)
        
        image_features = image_encoder.encode(image_array)
        if len(image_features.shape) > 1 and image_features.shape[0] == 1:
            image_features = image_features[0]
        
        if not args.no_normalize:
            image_features = normalize_features(image_features.reshape(1, -1))[0]
        else:
            image_features = image_features.flatten()
        
        print(f"  ✓ Image encoding completed: shape {image_features.shape}")
    except Exception as e:
        print(f"  ✗ Image encoding failed: {e}")
        sys.exit(1)
    
    # [5/5] Calculate similarity and display results
    print("\n[5/5] Calculating similarity and displaying results")
    print("=" * 70)
    
    # Calculate similarity matrix
    similarity_scores = compute_similarity_matrix(text_features, image_features)
    
    # Display results
    print(f"\nImage: {os.path.basename(args.image)}")
    print(f"  Feature shape: {image_features.shape}")
    print(f"  Feature norm: {np.linalg.norm(image_features):.6f}")
    
    print(f"\nText-Image Similarity:")
    print("-" * 70)
    print(f"{'Text':<50} {'Similarity':>10}")
    print("-" * 70)
    
    # Sort by similarity
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    for idx in sorted_indices:
        text = args.texts[idx]
        score = similarity_scores[idx]
        print(f"{text:<50} {score:>10.4f}")
    
    print("-" * 70)
    print(f"\nMost similar text: \"{args.texts[sorted_indices[0]]}\" (similarity: {similarity_scores[sorted_indices[0]]:.4f})")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()

