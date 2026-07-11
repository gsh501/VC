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
    def __init__(self, path="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.path = path
        self.image_input_list, self.image_ref_list = self.get_single_vimeo(filefolderlist=self.path)
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.image_input_list))

    def singleframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_single_vimeo(filefolderlist=self.path)

    def threeframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_three_vimeo(filefolderlist=self.path)

    def fourframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_4_vimeo(filefolderlist=self.path)

    def sixframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_6_vimeo(filefolderlist=self.path)

    def twelveframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_12_vimeo()

    def twenty_fourframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_24_vimeo()

    def thirty_oneframeTrain(self):
        self.image_input_list, self.image_ref_list = self.get_31_vimeo()


    def get_single_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sequences/",
                         filefolderlist="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])
            refnumber = curnumber - 1
            if refnumber < 1:
                continue

            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]
            for number in [0]:
                refnumber = curnumber + number
                refname = y[0:-5] + str(refnumber) + '.png'
                input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref 

    def get_three_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sequences/",
                         filefolderlist="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])
            refnumber = curnumber - 2
            if refnumber < 1:
                continue

            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]
            for number in [-1, 0]:
                refnumber = curnumber + number
                refname = y[0:-5] + str(refnumber) + '.png'
                input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref

    def get_4_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sequences/",
                       filefolderlist="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])
            refnumber = curnumber - 4
            if refnumber < 1:
                continue

            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]
            for number in [-3, -2, -1, 0]:
                refnumber = curnumber + number
                refname = y[0:-5] + str(refnumber) + '.png'
                input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref

    def get_6_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sequences/",
                        filefolderlist="/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/train.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            curnumber = int(y[-5:-4])
            refnumber = curnumber - 6
            if refnumber < 1:
                continue

            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

            for number in [-5, -4, -3, -2, -1, 0]:
                refnumber = curnumber + number
                refname = y[0:-5] + str(refnumber) + '.png'
                input_list += [refname]

            # for refnumber in range(curnumber - 1, curnumber - 7, -1):
            #     refname = y[0:-5] + str(refnumber) + '.png'
            #     input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref
    

    # /media/dongnan/1bd6ccf9-7315-4a88-8334-a9ee8489a977/wzq/data/GOT-10k/train
    # /media/dongnan/1bd6ccf9-7315-4a88-8334-a9ee8489a977/wzq/data/GOT-10k/out.txt

    def get_12_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data_long",
                        filefolderlist="/home/serverdn/hdd-0/slf_data/data_long/output_paths_no_transitions.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            
            # Extract the current image number (keeping it as a string)
            curnumber = int(y[-7:-4])  # Extract the numeric part, e.g., '002' -> 2

            # The current image is the reference image
            refname = y[0:-7] + str(curnumber).zfill(3) + '.png'
            fns_train_ref += [refname]

            # Collect the next 31 images as input images (from curnumber + 1 to curnumber + 31)
            for number in range(1, 13):  # Starting from the next image
                refnumber = curnumber + number
                refname = y[0:-7] + str(refnumber).zfill(3) + '.png'  # Zero-pad the number to 3 digits
                input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref
    
    def get_24_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data_long",
                        filefolderlist="/home/serverdn/hdd-0/slf_data/data_long/output_paths_no_transitions.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            
            # Extract the current image number (keeping it as a string)
            curnumber = int(y[-7:-4])  # Extract the numeric part, e.g., '002' -> 2

            # The current image is the reference image
            refname = y[0:-7] + str(curnumber).zfill(3) + '.png'
            fns_train_ref += [refname]

            # Collect the next 31 images as input images (from curnumber + 1 to curnumber + 31)
            for number in range(1, 25):  # Starting from the next image
                refnumber = curnumber + number
                refname = y[0:-7] + str(refnumber).zfill(3) + '.png'  # Zero-pad the number to 3 digits
                input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref
    
    def get_31_vimeo(self, rootdir="/home/serverdn/hdd-0/slf_data/data_long",
                        filefolderlist="/home/serverdn/hdd-0/slf_data/data_long/output_paths_no_transitions.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            input_list = []

            y = os.path.join(rootdir, line.rstrip())
            
            # Extract the current image number (keeping it as a string)
            curnumber = int(y[-7:-4])  # Extract the numeric part, e.g., '002' -> 2

            # The current image is the reference image
            refname = y[0:-7] + str(curnumber).zfill(3) + '.png'
            fns_train_ref += [refname]

            # Collect the next 31 images as input images (from curnumber + 1 to curnumber + 31)
            for number in range(1, 32):  # Starting from the next image
                refnumber = curnumber + number
                refname = y[0:-7] + str(refnumber).zfill(3) + '.png'  # Zero-pad the number to 3 digits
                input_list += [refname]

            fns_train_input += [input_list]

        return fns_train_input, fns_train_ref

    

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.image_ref_list[index])
        ref_image = ref_image.astype(np.float32) / 255.0
        ref_image = ref_image.transpose(2, 0, 1)
        ref_image = torch.from_numpy(ref_image).float()
        ref_image = rgb2ycbcr(ref_image, is_bgr=False)
        # ref_image = ref_image.unsqueeze(0)  # 变成 [1, 3, H, W]
        # ref_image_y,ref_image_uv = yuv_444_to_420(ref_image)
        
        # ref_image = yuv_420_to_444(ref_image_y,ref_image_uv)
        # ref_image = ref_image.squeeze(0)
        
        input_images = []

        for input_name in self.image_input_list[index]:
            input_image = imageio.imread(input_name)
            input_image = input_image.astype(np.float32) / 255.0
            input_image = input_image.transpose(2, 0, 1)
            input_image = torch.from_numpy(input_image).float()
            input_image = rgb2ycbcr(input_image, is_bgr=False)
            # input_image = input_image.unsqueeze(0)   # 变成 [1, 3, H, W]
            # input_image_y,input_image_uv = yuv_444_to_420(input_image)
            # input_image = yuv_420_to_444(input_image_y,input_image_uv)

            # input_image = input_image.squeeze(0)
            input_images.append(input_image)

                
        input_images = torch.cat(input_images, 0)

        # print(input_images.size(0))  # 调用size方法并传入维度索引
        
        if input_images.size(0)<30:
            ref_image, input_images = random_crop_and_pad_image_and_labels(ref_image, input_images,
                                                                            [self.im_height, self.im_width])
            ref_image, input_images = random_flip(ref_image, input_images)
        else:
            # print(input_images.shape)
            # ref_image = ref_image.unsqueeze(0) 
            # input_images = input_images.unsqueeze(0)
            # ref_image = F.interpolate(ref_image, size=(448, 256), mode='bilinear', align_corners=False)
            # input_images = F.interpolate(input_images, size=(448, 256), mode='bilinear', align_corners=False)
            # ref_image = ref_image.squeeze(0)
            # input_images = input_images.squeeze(0)
            ref_image, input_images = random_crop_and_pad_image_and_labels(ref_image, input_images,
                                                                            [self.im_height, self.im_width])
            ref_image, input_images = random_flip(ref_image, input_images)
            # print(input_images.shape)
        
        return ref_image, input_images


