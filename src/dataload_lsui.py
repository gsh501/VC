import os
import torch
import imageio
import numpy as np
import torch.utils.data as data
import random
import torch.nn.functional as F
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, yuv_420_to_444


def random_crop_and_pad_image_and_labels(image, labels, size):
    """Crop and pad image and labels to the specified size."""
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 
                                     0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])


def random_flip(images, labels):
    """Random horizontal and vertical flip."""
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels


class LSUIDataSet(data.Dataset):
    """
    LSUI Dataset loader for input/GT pairs.
    
    Dataset structure:
    - root/input/  : Input images (degraded/underwater)
    - root/GT/     : Ground truth images (clean/enhanced)
    
    Each single image is replicated to form a GOP (since LSUI contains single images, not sequences).
    """
    def __init__(self, root="/home/admin1/Data/water_enhance/LSUI", 
                 im_height=256, im_width=256, 
                 frame_count=3,  # Fixed to 3 frames per GOP
                 train=True,
                 max_samples=None):
        """
        Args:
            root: Root directory containing 'input' and 'GT' folders
            im_height: Height for random crop
            im_width: Width for random crop
            frame_count: Number of frames per GOP (each image will be replicated this many times)
            train: If True, use training set (input/GT), else use test set (test_input/test_gt)
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.root = root
        self.im_height = im_height
        self.im_width = im_width
        self.frame_count = frame_count
        self.train = train
        
        # Set input and GT directories
        if train:
            self.input_dir = os.path.join(root, "input")
            self.gt_dir = os.path.join(root, "GT")
        else:
            self.input_dir = os.path.join(root, "test_input")
            self.gt_dir = os.path.join(root, "test_gt")
        
        # Get all image files (each image will be used as a separate sample)
        input_files = sorted(os.listdir(self.input_dir), 
                            key=lambda x: int(x.split('.')[0]))
        
        if max_samples is not None:
            input_files = input_files[:max_samples]
        
        # Build image pair list (single images, not GOPs)
        self.image_input_list = []  # List of input image paths
        self.image_gt_list = []     # List of GT image paths
        
        for filename in input_files:
            input_path = os.path.join(self.input_dir, filename)
            gt_path = os.path.join(self.gt_dir, filename)
            
            # Check if both files exist
            if os.path.exists(input_path) and os.path.exists(gt_path):
                self.image_input_list.append(input_path)
                self.image_gt_list.append(gt_path)
        
        print(f"LSUI Dataset ({'train' if train else 'test'}): "
              f"Found {len(self.image_input_list)} image pairs, each will be replicated {frame_count} times")
    
    def __len__(self):
        return len(self.image_input_list)
    
    def __getitem__(self, index):
        """
        Returns:
            ref_image: Reference GT frame [3, H, W]
            input_images: Input frames replicated frame_count times [frame_count*3, H, W]
            gt_images: GT frames replicated frame_count times [frame_count*3, H, W]
        """
        # Load single GT image
        gt_path = self.image_gt_list[index]
        gt_image = imageio.imread(gt_path)
        gt_image = gt_image.astype(np.float32) / 255.0
        gt_image = gt_image.transpose(2, 0, 1)  # HWC -> CHW
        gt_image = torch.from_numpy(gt_image).float()
        gt_image = rgb2ycbcr(gt_image, is_bgr=False)
        
        # Load single input image
        input_path = self.image_input_list[index]
        input_image = imageio.imread(input_path)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)  # HWC -> CHW
        input_image = torch.from_numpy(input_image).float()
        input_image = rgb2ycbcr(input_image, is_bgr=False)
        
        if self.train:
            # Apply augmentation (crop and flip) to both images
            # Concatenate input and GT for consistent cropping
            combined = torch.cat([input_image, gt_image], 0)  # [6, H, W]
            
            # Use GT as reference for cropping
            ref_image, combined = random_crop_and_pad_image_and_labels(
                gt_image, 
                combined, 
                [self.im_height, self.im_width]
            )
            ref_image, combined = random_flip(ref_image, combined)
            
            # Split back
            input_image = combined[:3]
            gt_image = combined[3:]
        else:
            ref_image = gt_image
        
        # Replicate the images frame_count times to form a GOP
        input_images = torch.cat([input_image] * self.frame_count, 0)  # [frame_count*3, H, W]
        gt_images = torch.cat([gt_image] * self.frame_count, 0)  # [frame_count*3, H, W]
        
        return ref_image, input_images, gt_images


class LSUITestDataSet(data.Dataset):
    """
    LSUI Test Dataset for GOP-based evaluation.
    Each single image is replicated to form a GOP (since LSUI contains single images).
    """
    def __init__(self, root="/home/admin1/Data/water_enhance/LSUI", 
                 gop=3,  # Fixed to 3 frames per GOP (replicated from single image)
                 testfull=True,
                 train=False):
        """
        Args:
            root: Root directory containing test_input and test_gt folders
            gop: Group of pictures size (each image will be replicated this many times)
            testfull: If True, test all images; if False, test only first image
            train: If True, use train set; if False, use test set
        """
        self.root = root
        self.gop = gop
        
        # Set directories
        if train:
            self.input_dir = os.path.join(root, "input")
            self.gt_dir = os.path.join(root, "GT")
        else:
            self.input_dir = os.path.join(root, "test_input")
            self.gt_dir = os.path.join(root, "test_gt")
        
        # Get all image files (each will be a separate sample)
        imlist = sorted(os.listdir(self.input_dir), 
                       key=lambda x: int(x.split('.')[0]))
        
        self.input_list = []  # List of input image paths
        self.gt_list = []     # List of GT image paths
        self.image_names = []  # Filenames
        
        cnt = len(imlist)
        
        # Determine how many samples to use
        num_samples = cnt if testfull else 1
        
        for i in range(num_samples):
            filename = imlist[i]
            input_path = os.path.join(self.input_dir, filename)
            gt_path = os.path.join(self.gt_dir, filename)
            
            if os.path.exists(input_path) and os.path.exists(gt_path):
                self.input_list.append(input_path)
                self.gt_list.append(gt_path)
                self.image_names.append(filename)
        
        print(f"LSUI Test Dataset: Found {len(self.input_list)} images, each will be replicated {gop} times")
    
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, index):
        """
        Returns:
            input_images: Replicated input images [gop, 3, H, W]
            gt_images: Replicated GT images [gop, 3, H, W]
            image_name: Filename
        """
        # Load single input image
        input_path = self.input_list[index]
        input_image = imageio.imread(input_path).transpose(2, 0, 1).astype(np.float32) / 255.0
        input_image = torch.from_numpy(input_image).float()
        input_image = rgb2ycbcr(input_image)
        
        # Load single GT image
        gt_path = self.gt_list[index]
        gt_image = imageio.imread(gt_path).transpose(2, 0, 1).astype(np.float32) / 255.0
        gt_image = torch.from_numpy(gt_image).float()
        gt_image = rgb2ycbcr(gt_image)
        
        # Crop to dimensions divisible by 16 (better alignment for neural networks)
        _, h, w = input_image.shape
        h_aligned = (h // 16) * 16
        w_aligned = (w // 16) * 16
        input_image = input_image[:, :h_aligned, :w_aligned]
        gt_image = gt_image[:, :h_aligned, :w_aligned]
        
        # Apply YUV 420 conversion
        input_image = input_image.unsqueeze(0)  # [1, 3, H, W]
        input_image_y, input_image_uv = yuv_444_to_420(input_image)
        input_image = yuv_420_to_444(input_image_y, input_image_uv)
        input_image = input_image.squeeze(0)  # [3, H, W]
        
        gt_image = gt_image.unsqueeze(0)  # [1, 3, H, W]
        gt_image_y, gt_image_uv = yuv_444_to_420(gt_image)
        gt_image = yuv_420_to_444(gt_image_y, gt_image_uv)
        gt_image = gt_image.squeeze(0)  # [3, H, W]
        
        # Replicate the images gop times to form a GOP
        input_images = input_image.unsqueeze(0).repeat(self.gop, 1, 1, 1)  # [gop, 3, H, W]
        gt_images = gt_image.unsqueeze(0).repeat(self.gop, 1, 1, 1)  # [gop, 3, H, W]
        
        return input_images, gt_images, self.image_names[index]

