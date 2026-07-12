import os
import torch
import random
import re
import imageio.v2 as imageio
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from src_uf.utils.transforms import rgb2ycbcr, yuv_444_to_420, yuv_420_to_444
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


#将(C,T,H,W)转化为(C*T,H,W),然后进行随机裁剪和填充,最后再转化为(C,T,H,W)
def random_crop_and_pad_chunk_sequence(chunks, size):
    """Apply one random crop to all chunks in a GOP."""
    chunk_num, chunk_size, channels, _, _ = chunks.shape
    flat = chunks.reshape(chunk_num * chunk_size * channels, chunks.shape[-2], chunks.shape[-1])
    image_shape = flat.size()
    target_h, target_w = size

    pad_w = max(target_w, image_shape[2]) - image_shape[2]
    pad_h = max(target_h, image_shape[1]) - image_shape[1]
    flat = F.pad(flat, (0, pad_w, 0, pad_h))

    max_h = max(target_h, image_shape[1])
    max_w = max(target_w, image_shape[2])
    free_h = random.randint(0, max_h - target_h)
    free_w = random.randint(0, max_w - target_w)

    flat = flat[:, free_h:free_h + target_h, free_w:free_w + target_w]
    return flat.reshape(chunk_num, chunk_size, channels, target_h, target_w)

def random_flip_chunk_sequence(chunks):
    """Apply one random flip to all chunks in a GOP."""
    if random.randint(0, 1) == 1:
        chunks = torch.flip(chunks, [3])
    if random.randint(0, 1) == 1:
        chunks = torch.flip(chunks, [4])
    return chunks


#读取txt文件，该文件里记录着所有需要训练的图像路径
def _read_filelist(filelist):
    if filelist is None:
        raise ValueError("filelist must be provided.")
    with open(filelist) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]

def _load_image(path, convert_ycbcr=True):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float()

    if convert_ycbcr:
        image = rgb2ycbcr(image, is_bgr=False)
        image = image.unsqueeze(0)
        image_y, image_uv = yuv_444_to_420(image)
        image = yuv_420_to_444(image_y, image_uv)
        image = image.squeeze(0)
    return image

def _crop_to_common_size(frames, align=None):
    min_h = min(frame.shape[-2] for frame in frames)
    min_w = min(frame.shape[-1] for frame in frames)

    if align is not None:
        min_h = (min_h // align) * align
        min_w = (min_w // align) * align

    if min_h <= 0 or min_w <= 0:
        raise ValueError("Image size is too small after alignment.")

    return [frame[:, :min_h, :min_w] for frame in frames]

def _load_chunk(paths, convert_ycbcr=True, align=None):
    frames = [_load_image(path, convert_ycbcr=convert_ycbcr) for path in paths]
    frames = _crop_to_common_size(frames, align=align)
    return torch.stack(frames, 0)

def _resolve_path(rootdir, line):
    line = line.strip()
    if os.path.isabs(line) or rootdir is None:
        return line
    return os.path.join(rootdir, line)

def _split_numbered_path(path):
    """
    Split paths like /seq/im001.png into prefix, number, width, suffix.

    This replaces the hard-coded y[-7:-4] / y[-5:-4] logic used by the RT loader.
    """
    directory, filename = os.path.split(path)
    match = re.compile(r"^(.*?)(\d+)(\.[^.]+)$").match(filename)
    if match is None:
        raise ValueError(f"Cannot find a numeric frame index in path: {path}")

    name_prefix, frame_number, suffix = match.groups()
    prefix = os.path.join(directory, name_prefix)
    return prefix, int(frame_number), len(frame_number), suffix



def _numeric_sort_key(path):
    try:
        _, frame_number, _, _ = _split_numbered_path(path.strip())
        return frame_number
    except ValueError:
        return path.strip()



#训练使用，至少有8帧。 dataset.set_gop(8N)
#输出：ref_chunk    [8, 3, H, W]    input_chunks [N-1, 8, 3, H, W]
class UFDataSet(data.Dataset):
    def __init__(
        self,
        path=None,
        rootdir=None,
        filefolderlist=None,
        im_height=256,
        im_width=256,
        chunk_size=8,
        gop=32,
        convert_ycbcr=True,
        check_exists=False,
        max_samples=None,
        pad_last=True,
    ):
        self.path = path
        self.rootdir = rootdir
        self.filefolderlist = filefolderlist or path
        if self.filefolderlist is None:
            raise ValueError("UFDataSet requires filefolderlist or path. Pass it from the training script.")
        self.im_height = im_height
        self.im_width = im_width
        self.chunk_size = chunk_size
        self.gop = self._normalize_gop(gop)
        self.convert_ycbcr = convert_ycbcr
        self.check_exists = check_exists
        self.max_samples = max_samples
        self.pad_last = pad_last

        self.gops = self.get_uf_gops(
            self.rootdir,
            self.filefolderlist,
            self.chunk_size,
            self.gop,
        )
        # print(f"UF Dataset found GOPs: {len(self.gops)}")

    def set_frame_count(self, frame_count):
        self.set_gop(frame_count)

    def set_gop(self, gop):
        self.gop = self._normalize_gop(gop)
        self.gops = self.get_uf_gops(
            self.rootdir,
            self.filefolderlist,
            self.chunk_size,
            self.gop,
        )

    def _normalize_gop(self, gop):
        if gop < self.chunk_size:
            raise ValueError(f"gop must be >= chunk_size ({self.chunk_size}), got {gop}")
        return ((gop + self.chunk_size - 1) // self.chunk_size) * self.chunk_size

    def get_uf_gops(self, rootdir, filefolderlist, chunk_size, gop):
        frame_paths = [_resolve_path(rootdir, line) for line in _read_filelist(filefolderlist)]
        frame_paths = sorted(frame_paths, key=self._sequence_sort_key)

        sequences = []
        current_key = None
        current_paths = []
        for path in frame_paths:
            key = self._sequence_key(path)
            if current_key is not None and key != current_key:
                sequences.append(current_paths)
                current_paths = []
            current_key = key
            current_paths.append(path)
        if current_paths:
            sequences.append(current_paths)

        gops = []
        for sequence in sequences:
            for start in range(0, len(sequence), gop):
                paths = sequence[start:start + gop]
                if len(paths) < chunk_size and not self.pad_last:
                    continue

                paths = self._pad_to_chunk_multiple(paths, chunk_size)
                if len(paths) < chunk_size:
                    continue

                if self.check_exists and not all(os.path.exists(path) for path in paths):
                    continue

                gops.append(paths)

        if self.max_samples is not None:
            gops = gops[:self.max_samples]

        return gops

    @staticmethod
    def _sequence_key(path):
        prefix, _, _, suffix = _split_numbered_path(path)
        return prefix, suffix

    @staticmethod
    def _sequence_sort_key(path):
        prefix, frame_number, _, suffix = _split_numbered_path(path)
        return prefix, suffix, frame_number

    def _pad_to_chunk_multiple(self, paths, chunk_size):
        if not paths:
            return paths

        if not self.pad_last:
            valid_len = (len(paths) // chunk_size) * chunk_size
            return paths[:valid_len]

        target_len = ((len(paths) + chunk_size - 1) // chunk_size) * chunk_size
        while len(paths) < target_len:
            paths.append(paths[-1])
        return paths

    def __len__(self):
        return len(self.gops)

    def __getitem__(self, index):
        paths = self.gops[index]
        chunks = []
        for start in range(0, len(paths), self.chunk_size):
            chunk_paths = paths[start:start + self.chunk_size]
            if len(chunk_paths) < self.chunk_size:
                continue
            chunks.append(_load_chunk(chunk_paths, convert_ycbcr=self.convert_ycbcr))

        chunks = torch.stack(chunks, 0)
        chunks = random_crop_and_pad_chunk_sequence(chunks, [self.im_height, self.im_width])
        chunks = random_flip_chunk_sequence(chunks)

        ref_chunk = chunks[0]
        input_chunks = chunks[1:]
        return ref_chunk, input_chunks


# 按gop（32）划分，每个gop输出 ref_chunk ：[8, 3, H, W]  input_chunks [3, 8, 3, H, W]
class UFTestDataSet(data.Dataset):
    """
    UF test loader for GOP evaluation.

    It groups filelist entries exactly like the RT TetsDataSet, then splits each
    GOP into fixed-size chunks:
        ref_chunk: [chunk_size, 3, H, W]
        input_chunks: [num_chunks - 1, chunk_size, 3, H, W]
        image_names: list[str]
    """
    def __init__(self, root=None, filelist=None, gop=32, chunk_size=8, testfull=True, convert_ycbcr=True, align=8, pad_last=True):
        self.root = root
        self.filelist = filelist
        self.gop = gop
        self.chunk_size = chunk_size
        self.convert_ycbcr = convert_ycbcr
        self.align = align
        self.pad_last = pad_last
        self.image_names = []

        imlist = _read_filelist(filelist)
        imlist = sorted(imlist, key=_numeric_sort_key)

        cnt = len(imlist)

        if testfull:
            gop_count = cnt // self.gop
            if cnt % self.gop > 0:
                gop_count += 1
        else:
            gop_count = 1

        self.gops = []

        for i in range(gop_count):
            start = i * self.gop
            end = min(start + self.gop, cnt)
            names = imlist[start:end]
            if not names:
                continue

            paths = [_resolve_path(root, name) for name in names]
            if self.pad_last:
                target_len = ((len(paths) + chunk_size - 1) // chunk_size) * chunk_size
                while len(paths) < target_len:
                    paths.append(paths[-1])
                    names.append(names[-1])

            if len(paths) < chunk_size:
                continue

            self.gops.append(paths)   #gop中图像的路径
            self.image_names.append(names)      #gop图像的命名

    def __len__(self):
        return len(self.gops)

    def __getitem__(self, index):
        paths = self.gops[index]
        chunks = []
        for start in range(0, len(paths), self.chunk_size):
            chunk_paths = paths[start:start + self.chunk_size]
            if len(chunk_paths) < self.chunk_size:
                continue
            chunks.append(_load_chunk(chunk_paths, convert_ycbcr=self.convert_ycbcr, align=self.align))

        chunks = torch.stack(chunks, 0)
        ref_chunk = chunks[0]
        input_chunks = chunks[1:]
        return ref_chunk, input_chunks, self.image_names[index]
