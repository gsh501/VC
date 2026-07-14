#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to view saved test images from training
"""

import os
import argparse
import glob
from PIL import Image
import matplotlib.pyplot as plt


def view_test_results(base_dir, epoch, model_name="DMC_slf_yuv420_lsui", quality_level=2):
    """
    View test images for a specific epoch
    
    Args:
        base_dir: Base directory containing pretrained models
        epoch: Epoch number to view
        model_name: Model name
        quality_level: Quality level
    """
    # Construct path to test images
    test_images_dir = os.path.join(base_dir, model_name, str(quality_level), f"test_images_epoch_{epoch}")
    
    if not os.path.exists(test_images_dir):
        print(f"Error: Directory not found: {test_images_dir}")
        return
    
    # Find all image sets
    input_images = sorted(glob.glob(os.path.join(test_images_dir, "*_input.png")))
    
    if len(input_images) == 0:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(input_images)} test image sets in epoch {epoch}")
    
    for input_path in input_images:
        base_name = os.path.basename(input_path).replace("_input.png", "")
        gt_path = os.path.join(test_images_dir, f"{base_name}_gt.png")
        output_path = os.path.join(test_images_dir, f"{base_name}_output.png")
        
        if not os.path.exists(gt_path) or not os.path.exists(output_path):
            print(f"Warning: Missing images for {base_name}")
            continue
        
        # Load images
        input_img = Image.open(input_path)
        gt_img = Image.open(gt_path)
        output_img = Image.open(output_path)
        
        # Display images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(input_img)
        axes[0].set_title(f"Input (Degraded)\n{base_name}")
        axes[0].axis('off')
        
        axes[1].imshow(gt_img)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(output_img)
        axes[2].set_title("Model Output")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(test_images_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved comparison: {base_name}_comparison.png")


def list_available_epochs(base_dir, model_name="DMC_slf_yuv420_lsui", quality_level=2):
    """List all available epoch results"""
    model_dir = os.path.join(base_dir, model_name, str(quality_level))
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return []
    
    test_dirs = sorted(glob.glob(os.path.join(model_dir, "test_images_epoch_*")))
    epochs = []
    
    for test_dir in test_dirs:
        dir_name = os.path.basename(test_dir)
        epoch_str = dir_name.replace("test_images_epoch_", "")
        try:
            epoch = int(epoch_str)
            epochs.append(epoch)
            num_images = len(glob.glob(os.path.join(test_dir, "*_input.png")))
            print(f"Epoch {epoch:3d}: {num_images} test images")
        except ValueError:
            continue
    
    return epochs


def main():
    parser = argparse.ArgumentParser(description="View test results from training")
    parser.add_argument("--base_dir", default="./pretrained", help="Base directory for pretrained models")
    parser.add_argument("--model", default="DMC_slf_yuv420_lsui", help="Model name")
    parser.add_argument("--quality_level", type=int, default=2, help="Quality level")
    parser.add_argument("--epoch", type=int, help="Specific epoch to view (optional)")
    parser.add_argument("--list", action="store_true", help="List all available epochs")
    args = parser.parse_args()
    
    if args.list or args.epoch is None:
        print(f"\nAvailable test results in {args.base_dir}/{args.model}/{args.quality_level}:")
        print("-" * 50)
        epochs = list_available_epochs(args.base_dir, args.model, args.quality_level)
        
        if len(epochs) == 0:
            print("No test results found yet.")
        else:
            print(f"\nTotal: {len(epochs)} epoch(s) with test images")
            if args.epoch is None:
                print("\nUse --epoch <N> to view specific epoch results")
                return
    
    if args.epoch is not None:
        print(f"\nViewing test results for epoch {args.epoch}...")
        print("=" * 50)
        view_test_results(args.base_dir, args.epoch, args.model, args.quality_level)


if __name__ == "__main__":
    main()

