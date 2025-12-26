"""
CLIP Image Encoder DXNN Test - Simple CLI version
Simple educational code for testing DEEPX NPU-based CLIP image encoder
Based on DEEPX-AI/dx_clip_demo structure
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from dx_engine import InferenceEngine
import torch
from torchvision import transforms


class ImageEncoder:
    """Image encoder using DEEPX NPU (DXNN)"""
    
    def __init__(self, dxnn_path: str):
        if not os.path.isfile(dxnn_path):
            raise FileNotFoundError(f"DXNN model file not found: {dxnn_path}")
        
        self.engine = InferenceEngine(dxnn_path)
        self.input_info = self.engine.get_input_tensors_info()
        print(f"  Model input info: {self.input_info}")
    
    def encode(self, image_array: np.ndarray):
        """Encode image array to feature vector"""
        # Run inference on DEEPX NPU
        outputs = self.engine.run(image_array)
        
        # Get first output (feature vector)
        if isinstance(outputs, list):
            image_features = outputs[0]
        else:
            image_features = outputs
        
        return np.array(image_features)


def create_image_transform():
    """Create CLIP image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, 
                         max_size=None, antialias=True),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        # CLIP normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_and_preprocess_image(image_path: str, transform):
    """Load image and preprocess for CLIP"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        # Convert to numpy array and add batch dimension
        image_array = image_tensor.numpy()
        # Change from CHW to HWC if needed, or keep CHW based on model input
        # DXNN typically expects CHW format with batch dimension
        image_array = np.expand_dims(image_array, axis=0)  # [1, C, H, W]
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def normalize_features(features: np.ndarray):
    """L2 normalize feature vectors"""
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / norms


def main():
    parser = argparse.ArgumentParser(description="CLIP Image Encoder DXNN Test")
    parser.add_argument("--model", type=str,
                       default="./dxnn/ViT-L-14-quickgelu-dfn2b.dxnn",
                       help="Path to image encoder DXNN model")
    parser.add_argument("--image", type=str, required=True,
                       help="Image file path to encode")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Don't normalize output features")
    
    args = parser.parse_args()
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"Error: DXNN model file not found: {args.model}")
        sys.exit(1)
    
    # Check image file
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    print("=" * 60)
    print("CLIP Image Encoder DXNN Test")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    try:
        image_encoder = ImageEncoder(args.model)
        print(f"  ✓ Model loaded: {args.model}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Prepare input
    print(f"\n[2/4] Preparing input...")
    transform = create_image_transform()
    
    try:
        image_array = load_and_preprocess_image(args.image, transform)
        print(f"  ✓ Loaded: {args.image} (shape: {image_array.shape})")
    except Exception as e:
        print(f"  ✗ Failed to load {args.image}: {e}")
        sys.exit(1)
    
    # Run inference
    print(f"\n[3/4] Running inference...")
    try:
        image_features = image_encoder.encode(image_array)
        # Remove batch dimension if present
        if len(image_features.shape) > 1 and image_features.shape[0] == 1:
            image_features = image_features[0]
        print(f"  ✓ Inference completed: output shape {image_features.shape}")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        sys.exit(1)
    
    # Normalize
    if not args.no_normalize:
        image_features = normalize_features(image_features.reshape(1, -1))[0]
    
    # Display results
    print(f"\n[4/4] Results")
    print("=" * 60)
    
    print(f"\nImage: {os.path.basename(args.image)}")
    print(f"  Feature shape: {image_features.shape}")
    print(f"  Feature norm: {np.linalg.norm(image_features):.6f}")
    feature_values = image_features[:10]
    print(f"  First 10 values: {np.array2string(feature_values, precision=4, suppress_small=True)}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()


