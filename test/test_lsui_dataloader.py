#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for LSUI dataset loader
"""

import torch
from torch.utils.data import DataLoader
from src.dataload_lsui import LSUIDataSet, LSUITestDataSet


def test_train_dataset():
    """Test the training dataset loader"""
    print("=" * 80)
    print("Testing LSUI Training Dataset")
    print("=" * 80)
    
    dataset = LSUIDataSet(
        root="/home/admin1/Data/water_enhance/LSUI",
        im_height=256,
        im_width=256,
        frame_count=7,
        train=True,
        max_samples=100  # Only test with first 100 images
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    ref_image, input_images, gt_images = dataset[0]
    
    print(f"Reference image shape: {ref_image.shape}")
    print(f"Input images shape: {input_images.shape}")
    print(f"GT images shape: {gt_images.shape}")
    print(f"Reference image range: [{ref_image.min():.3f}, {ref_image.max():.3f}]")
    print(f"Input images range: [{input_images.min():.3f}, {input_images.max():.3f}]")
    print(f"GT images range: [{gt_images.min():.3f}, {gt_images.max():.3f}]")
    
    # Test with DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    ref_batch, input_batch, gt_batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"Reference batch: {ref_batch.shape}")
    print(f"Input batch: {input_batch.shape}")
    print(f"GT batch: {gt_batch.shape}")
    
    print("\n✓ Training dataset test passed!")


def test_test_dataset():
    """Test the test dataset loader"""
    print("\n" + "=" * 80)
    print("Testing LSUI Test Dataset")
    print("=" * 80)
    
    dataset = LSUITestDataSet(
        root="/home/admin1/Data/water_enhance/LSUI",
        gop=12,
        testfull=False,  # Only test first GOP
        train=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test first sample
        input_images, gt_images, image_names = dataset[0]
        
        print(f"Input images shape: {input_images.shape}")
        print(f"GT images shape: {gt_images.shape}")
        print(f"Number of images: {len(image_names)}")
        print(f"First few filenames: {image_names[:3]}")
        print(f"Input range: [{input_images.min():.3f}, {input_images.max():.3f}]")
        print(f"GT range: [{gt_images.min():.3f}, {gt_images.max():.3f}]")
        
        # Test with DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        input_batch, gt_batch, names = next(iter(loader))
        
        print(f"\nBatch shapes:")
        print(f"Input batch: {input_batch.shape}")
        print(f"GT batch: {gt_batch.shape}")
        
        print("\n✓ Test dataset test passed!")
    else:
        print("⚠ Warning: No test samples found!")


def main():
    print("\nLSUI Dataset Loader Test\n")
    
    try:
        test_train_dataset()
        test_test_dataset()
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

