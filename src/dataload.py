import os
import torch
import imageio
import numpy as np
import torch.utils.data as data
# from subnet.basics import *
import random
import torch.nn.functional as F
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, yuv_420_to_444
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import imageio
import torchvision.transforms as transforms

def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])
    # freesize0 = freesize1 = 0
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])

def random_flip(images, labels):

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





class TetsDataSet(data.Dataset):
    def __init__(self, root=None, filelist=None,gop=12, testfull=True):
        """
        root: Dataset root directory.
        filelist: List of sequences to use.
        gop: User-defined group of pictures (number of frames per group).
        testfull: Whether to test the full sequence or just 1 frame.
        """
        with open(filelist) as f:
            imlist = f.readlines()

        self.ref = []   # Reference frames (now the first frame of each GOP).
        self.input = [] # Input image sequences.
        self.gop = gop
        self.image_names = []  # To store the filenames

        # Sort image filenames numerically based on the numeric part of the filename
        imlist = sorted(imlist, key=lambda x: int(x.strip().split('.')[0][2:]))

        cnt = len(imlist)

        if testfull:
            framerange = cnt // self.gop
            if cnt % self.gop > 0:
                framerange += 1
        else:
            framerange = 1

        for i in range(framerange):
            inputpath = []
            filenames = []
            for j in range(self.gop):
                img_index = i * self.gop + j
                if img_index < cnt:
                    img_path = os.path.join(root, imlist[img_index].strip())
                    inputpath.append(img_path)
                    filenames.append(imlist[img_index].strip())  # Store the filenames
                else:
                    break
            if len(inputpath) > 0:
                self.input.append(inputpath)
                self.ref.append(inputpath[0])  # Use the first image of the group as reference
                self.image_names.append(filenames)  # Store the filenames for the group

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        
        for filename in self.input[index]:
            input_image = imageio.imread(filename).transpose(2, 0, 1).astype(np.float32) / 255.0
            # 如果需要裁剪为64的倍数，取消以下注释
            # h = (input_image.shape[1] // 64) * 64
            # w = (input_image.shape[2] // 64) * 64
            # input_image = input_image[:, :h, :w]
            
            input_image = torch.from_numpy(input_image).float()

            # 将 input_image 转换为 YCbCr 空间
            input_image = rgb2ycbcr(input_image)
            input_image = input_image.unsqueeze(0)   # 变成 [1, 3, H, W]
            input_image_y,input_image_uv = yuv_444_to_420(input_image)
            # print(input_image.shape)
            input_image = yuv_420_to_444(input_image_y,input_image_uv)

            input_image = input_image.squeeze(0)

            input_images.append(input_image)

        input_images = torch.stack(input_images, 0)

        # 处理参考图像
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        ref_image = torch.from_numpy(ref_image).float()
        

        # 将参考图像转换为 YCbCr 空间
        ref_image = rgb2ycbcr(ref_image)
        ref_image = ref_image.unsqueeze(0)  # 变成 [1, 3, H, W]
        ref_image_y,ref_image_uv = yuv_444_to_420(ref_image)
        
        ref_image = yuv_420_to_444(ref_image_y,ref_image_uv)
        ref_image = ref_image.squeeze(0)
        

        # 返回图像和文件名
        return ref_image, input_images, self.image_names



class DataSet(data.Dataset):
    def __init__(self, path=None, im_height=256, im_width=256):
        self.path = path
        self.im_height = im_height
        self.im_width = im_width
        self.frame_count = 1 # Default frame count
        self.image_input_list, self.image_ref_list = self.get_vimeo_frames(self.frame_count)
        print(f"Dataset found images: {len(self.image_input_list)}")

    def set_frame_count(self, frame_count):
        """ Set the frame count for the dataset. """
        self.frame_count = frame_count
        self.image_input_list, self.image_ref_list = self.get_vimeo_frames(frame_count)

    def get_vimeo_frames(self, frame_count):
        """ Load frames based on the specified frame count. """
        if frame_count < 8:
            return self.get_single_or_multi_frame_vimeo("/home/admin1/Data/data/vimeo_septuplet/sequences/", "/home/admin1/Data/data/vimeo_septuplet/test.txt", frame_count)
        else:
            return self.get_long_sequence_vimeo("/home/admin1/Data/data/", "/home/admin1/Data/data/output_paths_no_transitions.txt",frame_count)

    def get_single_or_multi_frame_vimeo(self, rootdir, filefolderlist, frame_count):
        """ General method for getting frames less than 8. """
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for line in data:
            input_list = []
            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])  # Extract the current frame number (e.g., '001' -> 1)

            # Reference frame is the frame count earlier
            refnumber = curnumber - frame_count + 1
            if refnumber < 1:
                continue
            refname = y[0:-5] + str(curnumber - frame_count+1)+ '.png'
            # print(f"Reference frame: {refname}")
            fns_train_ref.append(refname)

            # Collect input frames based on the number of frames
            for number in range(1, frame_count):
                refnumber = curnumber + number - frame_count+1   # Adjusting the formula to match your clarification
                
                refname = y[0:-5] + str(refnumber) + '.png'  # Zero-pad the number
                input_list.append(refname)

            fns_train_input.append(input_list)

        return fns_train_input, fns_train_ref


    def get_long_sequence_vimeo(self, rootdir, filefolderlist,num):
        """ Method to handle sequences with frames >= 8. """
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for line in data:
            input_list = []
            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-7:-4])  # Adjust as needed

            refname = y[0:-7] + str(curnumber).zfill(3) + '.png'
            fns_train_ref.append(refname)

            for number in range(1, num):  # Example for 31 frames
                refname = y[0:-7] + str(curnumber + number).zfill(3) + '.png'
                input_list.append(refname)


            fns_train_input.append(input_list)

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.image_ref_list[index])
        ref_image = ref_image.astype(np.float32) / 255.0
        ref_image = ref_image.transpose(2, 0, 1)
        ref_image = torch.from_numpy(ref_image).float()
        ref_image = rgb2ycbcr(ref_image, is_bgr=False)

        input_images = []
        # print(self.image_ref_list[index])
        # print(self.image_input_list[index])
        for input_name in self.image_input_list[index]:
            input_image = imageio.imread(input_name)
            input_image = input_image.astype(np.float32) / 255.0
            input_image = input_image.transpose(2, 0, 1)
            input_image = torch.from_numpy(input_image).float()
            input_image = rgb2ycbcr(input_image, is_bgr=False)
            input_images.append(input_image)

        input_images = torch.cat(input_images, 0)

      
        ref_image, input_images = random_crop_and_pad_image_and_labels(ref_image, input_images, [self.im_height, self.im_width])
        ref_image, input_images = random_flip(ref_image, input_images)
        
        # if input_images.size(0) > 4:
        #         all_imgs = torch.cat([ref_image.unsqueeze(0), input_images], dim=0)  # [N+1, C, H, W]
        #         perm = torch.randperm(all_imgs.size(0), device=all_imgs.device)
        #         shuffled = all_imgs[perm]

        #         # 随机后的第 0 个作为新的 ref，其余作为新的 input_images
        #         ref_image, input_images = shuffled[0], shuffled[1:]

        return ref_image, input_images
