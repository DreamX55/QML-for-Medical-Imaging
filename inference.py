#!/usr/bin/env python
"""
Inference Script for Hybrid Quantum-Classical Classifier.

Provides single image and batch inference with explanations.

Usage:
    python inference.py --checkpoint model.pt --image sample.jpg
    python inference.py --checkpoint model.pt --image_dir ./test_images --batch
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from data.transforms import get_val_transforms
from models.hybrid_model import create_model
from xai.gradcam import GradCAM
from xai.visualizations import plot_gradcam_overlay


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of images for batch inference')
    parser.add_argument('--output_dir', type=str, default='./outputs/inference',
                        help='Output directory')
    parser.add_argument('--explain', action='store_true',
                        help='Generate Grad-CAM explanations')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint.get('config', {})
    config = Config._from_dict(config_dict) if config_dict else Config()
    
    # Create model
    model = create_model(config, model_type='hybrid')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def predict_single(model, image_path: str, config: Config, device: str, explain: bool = False):
    """Predict on a single image."""
    # Load and preprocess
    transform = get_val_transforms(config.data)
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax().item()
        confidence = probs[0, pred_class].item()
    
    result = {
        'image': image_path,
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy().tolist(),
    }
    
    # Generate explanation if requested
    if explain:
        gradcam = GradCAM(model)
        heatmap = gradcam(input_tensor, pred_class)
        result['gradcam_heatmap'] = heatmap
    
    return result


def predict_batch(model, image_dir: str, config: Config, device: str, output_dir: str, explain: bool = False):
    """Predict on a directory of images."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in image_extensions]
    
    results = []
    for path in image_paths:
        print(f"Processing: {path.name}")
        result = predict_single(model, str(path), config, device, explain)
        results.append(result)
        
        if explain and 'gradcam_heatmap' in result:
            img = np.array(Image.open(str(path)).convert('RGB'))
            save_path = f"{output_dir}/{path.stem}_gradcam.png"
            plot_gradcam_overlay(img, result['gradcam_heatmap'], save_path=save_path)
    
    # Save results
    import json
    with open(f"{output_dir}/predictions.json", 'w') as f:
        # Remove non-serializable items
        serializable = [{k: v for k, v in r.items() if k != 'gradcam_heatmap'} for r in results]
        json.dump(serializable, f, indent=2)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    if args.image:
        result = predict_single(model, args.image, config, device, args.explain)
        print(f"\nPrediction: Class {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['probabilities']}")
        
        if args.explain:
            img = np.array(Image.open(args.image).convert('RGB'))
            save_path = f"{args.output_dir}/gradcam_explanation.png"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            plot_gradcam_overlay(img, result['gradcam_heatmap'], save_path=save_path)
            print(f"Explanation saved to {save_path}")
    
    elif args.image_dir:
        results = predict_batch(model, args.image_dir, config, device, args.output_dir, args.explain)
        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to {args.output_dir}/predictions.json")
    
    else:
        print("Error: Provide --image or --image_dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
