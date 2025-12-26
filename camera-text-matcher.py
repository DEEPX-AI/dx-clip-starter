"""
Real-time Camera Text Matcher using CLIP
Match predefined texts with real-time images from USB camera
"""

import torch
import onnxruntime
import os
import sys
import argparse
import numpy as np
import open_clip
import re
import cv2
from PIL import Image
from dx_engine import InferenceEngine
from torchvision import transforms
import time


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


def preprocess_frame(frame, transform):
    """Preprocess OpenCV frame to CLIP input format"""
    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # NumPy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # Apply transform
    image_tensor = transform(pil_image)
    # Convert to numpy and add batch dimension
    image_array = image_tensor.numpy()
    image_array = np.expand_dims(image_array, axis=0)  # [1, C, H, W]
    return image_array


def normalize_features(features):
    """L2 normalization"""
    if isinstance(features, torch.Tensor):
        return features / features.norm(dim=-1, keepdim=True)
    else:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / norms


def compute_similarity(text_features, image_features):
    """Calculate similarity between text and images"""
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.cpu().numpy()
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.cpu().numpy()
    
    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)
    
    # Calculate cosine similarity (already normalized, so just dot product)
    similarity = np.dot(text_features, image_features.T)  # [num_texts, 1]
    return similarity.squeeze()  # [num_texts]


def draw_text_on_frame(frame, text, similarity, position=(10, 30)):
    """Draw text and similarity on frame"""
    # Background box
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    
    # Display text and similarity
    display_text = f"{text} ({similarity:.3f})"
    (display_width, display_height), _ = cv2.getTextSize(
        display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    
    # Draw background box
    cv2.rectangle(
        frame,
        (position[0] - 5, position[1] - display_height - 5),
        (position[0] + display_width + 5, position[1] + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        display_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    return frame


def draw_top_texts_on_frame(frame, texts_with_scores, max_texts=3, position=(10, 40), line_spacing=40):
    """Draw top N texts with their similarity scores on frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    padding = 5
    
    # Calculate total height needed
    max_width = 0
    total_height = 0
    display_texts = []
    
    for i, (text, score) in enumerate(texts_with_scores[:max_texts]):
        display_text = f"{i+1}. {text} ({score:.3f})"
        display_texts.append(display_text)
        (text_width, text_height), baseline = cv2.getTextSize(
            display_text, font, font_scale, thickness
        )
        max_width = max(max_width, text_width)
        total_height += text_height + baseline + line_spacing
    
    # Draw background box for all texts
    if display_texts:
        #cv2.rectangle(
        #    frame,
        #    (position[0] - padding, position[1] - padding),
        #    (position[0] + max_width + padding, position[1] + total_height + padding),
        #    (0, 0, 0),
        #    -1
        #)
        
        # Draw each text
        y_offset = position[1]
        for i, display_text in enumerate(display_texts):
            # Use different colors for ranking
            if i == 0:
                color = (0, 255, 0)  # Green for 1st
            elif i == 1:
                color = (0, 255, 255)  # Yellow for 2nd
            else:
                color = (0, 165, 255)  # Orange for 3rd
            
            cv2.putText(
                frame,
                display_text,
                (position[0], y_offset),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )
            
            (_, text_height), baseline = cv2.getTextSize(
                display_text, font, font_scale, thickness
            )
            y_offset += text_height + baseline + line_spacing
    
    return frame


def setup_camera(device_path="/dev/video0", width=1280, height=720, fps=30):
    """Setup camera"""
    cap = cv2.VideoCapture(device_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device_path}")
    
    # Set MJPEG format
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Check actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera settings:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps:.2f}")
    print(f"  Format: MJPEG")
    
    return cap


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Camera Text Matcher using CLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera-text-matcher.py \\
    --text-encoder onnx/ViT-L-14-quickgelu-dfn2b-text.onnx \\
    --image-encoder dxnn/ViT-L-14-quickgelu-dfn2b.dxnn \\
    --texts "a cat" "a dog" "a person" "a car" \\
    --camera /dev/video0
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
                       help="Text list to match (space-separated)")
    parser.add_argument("--camera", type=str, default="/dev/video0",
                       help="Camera device path")
    parser.add_argument("--width", type=int, default=1280,
                       help="Camera resolution width")
    parser.add_argument("--height", type=int, default=720,
                       help="Camera resolution height")
    parser.add_argument("--fps", type=int, default=30,
                       help="Camera FPS")
    parser.add_argument("--skip-frames", type=int, default=2,
                       help="Frame interval to process (1=all frames, 2=skip 1 frame)")
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
    
    print("=" * 70)
    print("Real-time Camera Text Matcher")
    print("=" * 70)
    
    # Extract model name
    if args.model_name is None:
        args.model_name = extract_model_name_from_path(args.text_encoder)
        print(f"Auto-detected model name: {args.model_name}")
    
    # [1/4] Load text encoder
    print("\n[1/4] Loading text encoder...")
    try:
        text_encoder = TextEncoder(args.text_encoder)
        print(f"  ✓ Text encoder loaded")
    except Exception as e:
        print(f"  ✗ Text encoder loading failed: {e}")
        sys.exit(1)
    
    # [2/4] Load image encoder
    print("\n[2/4] Loading image encoder...")
    try:
        image_encoder = ImageEncoder(args.image_encoder)
        print(f"  ✓ Image encoder loaded")
    except Exception as e:
        print(f"  ✗ Image encoder loading failed: {e}")
        sys.exit(1)
    
    # [3/4] Load tokenizer and encode text
    print("\n[3/4] Loading tokenizer and encoding text...")
    try:
        tokenizer = open_clip.get_tokenizer(args.model_name)
        print(f"  ✓ Tokenizer loaded")
        
        # Tokenize and encode text
        print(f"  Text list: {args.texts}")
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
    
    # [4/4] Setup camera and start streaming
    print("\n[4/4] Setting up camera...")
    try:
        cap = setup_camera(args.camera, args.width, args.height, args.fps)
        print(f"  ✓ Camera setup completed: {args.camera}")
    except Exception as e:
        print(f"  ✗ Camera setup failed: {e}")
        sys.exit(1)
    
    # Image preprocessing transform
    transform = create_image_transform()
    
    print("\n" + "=" * 70)
    print("Streaming started (Press 'q' to quit)")
    print("=" * 70)
    
    frame_count = 0
    
    # Variables to store previous results (initial values)
    # Store top 3 texts with scores: list of (text, score) tuples
    last_top_texts = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Frame skip handling
            if frame_count % args.skip_frames != 0:
                # Display skipped frames (maintain previous results)
                if last_top_texts:
                    frame = draw_top_texts_on_frame(frame, last_top_texts, max_texts=3)

                cv2.imshow('Camera Text Matcher', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            
            try:
                image_array = preprocess_frame(frame, transform)
                image_features = image_encoder.encode(image_array)
                
                if len(image_features.shape) > 1 and image_features.shape[0] == 1:
                    image_features = image_features[0]
                
                if not args.no_normalize:
                    image_features = normalize_features(image_features.reshape(1, -1))[0]
                else:
                    image_features = image_features.flatten()
                
                # Calculate similarity
                similarity_scores = compute_similarity(text_features, image_features)
                
                # Get top 3 texts sorted by similarity (highest first)
                sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
                top_texts = [(args.texts[idx], similarity_scores[idx]) for idx in sorted_indices[:3]]
                
                # Update previous results
                last_top_texts = top_texts
                
                # Display results on frame
                frame = draw_top_texts_on_frame(frame, top_texts, max_texts=3)

                
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Maintain previous results even on error
                if last_top_texts:
                    frame = draw_top_texts_on_frame(frame, last_top_texts, max_texts=3)
                continue
            
            # Display on screen
            cv2.imshow('Camera Text Matcher', frame)
            
            # Quit with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\nStreaming ended")
        print("=" * 70)


if __name__ == "__main__":
    main()

