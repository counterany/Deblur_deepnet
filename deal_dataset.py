import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
import time
from os.path import join
import cv2


class DeblurDataset(Dataset):
    def __init__(self, path, data_format, frames, crop_size=(256, 256)):
        self.crop_width, self.crop_height = crop_size
        start = time.time()
        self.frames = frames
        self.future_frames_num = 2
        self.past_frames_num = 2
        self.data_format = 'RGB'
        self.width = 640
        self.height = 480
        self.crop_size = crop_size

        self.samples = self.get_samples_data_full_path(path, data_format)  # 每一个元素包含八个图片名称地址
        end = time.time()
        print(f"下载图像文件需要的时间：{end - start}")

    def get_samples_data_full_path(self, data_path, data_format):
        """
        获得一个列表，里边每个值包含一个子列表，每个子列表又包含八字典，每字典有一对路径
        :param data_path:
        :param data_format:
        :return:
        """
        samples = []
        records = dict()
        file_name = sorted(os.listdir(data_path), key=int)
        for seq in file_name:
            records[seq] = list()
            # 每个records[seq]存100帧对图像的路径
            for frame in range(100):
                sample = dict()
                sample['blur'] = join(data_path, seq, 'Blur', data_format, '{:08d}.{}'.format(frame, 'png'))
                sample['clear'] = join(data_path, seq, 'Sharp', data_format, '{:08d}.{}'.format(frame, 'png'))
                records[seq].append(sample)
        for seq_r in records.values():  # 有很多个列表，每个seq_r列表的长度为100，每个元素为一个字典
            img_len = len(seq_r)
            temp = img_len - (self.frames - 1)  # 计算长度是否满足
            if temp > 0:
                # 循环获得八帧的图片，一次只放弃一张
                for idx in range(temp):
                    samples.append(seq_r[idx: idx + self.frames])
            else:
                raise IndexError('长度不够')
        return samples

    def __getitem__(self, item):
        # 随机裁剪
        top = random.randint(0, self.height - self.crop_height)
        left = random.randint(0, self.width - self.crop_width)
        # 随机翻转
        random_mode = random.randint(0, 7)
        blur_images, clear_images = [], []
        for sample_dict in self.samples[item]:
            blur_image, clear_image = load_image_data(sample_dict, top, left, random_mode, self.crop_size)
            blur_images.append(blur_image)
            clear_images.append(clear_image)
        images = []
        for item in [blur_images, clear_images]:
            temp = torch.cat(item, dim=0)
            images.append(temp)
        return images

    def __len__(self):
        return len(self.samples)


def normalize_data(img_data):
    """
    归一化数据
    :param img_data:
    :return:
    """
    img = np.array(img_data, dtype='float32') / 255.0
    img = (img - np.min(img)) / np.maximum((np.max(img) - np.min(img)), 0.001)
    img = torch.Tensor(img)
    return img


def load_image_data(sample_path, top, left, mode, crop_size=(256, 256)):
    # 读取数据
    blur_image = cv2.imread(sample_path['blur'])
    clear_image = cv2.imread(sample_path['clear'])
    # 裁剪数据
    blur_image = blur_image[top:top + crop_size[0], left:left + crop_size[0]]
    clear_image = clear_image[top:top + crop_size[1], left:left + crop_size[1]]
    # 调整数据
    blur_image = data_agument(blur_image, mode)
    clear_image = data_agument(clear_image, mode)
    # 转化为tensor
    blur_image = torch.from_numpy(np.ascontiguousarray(blur_image.transpose((2, 0, 1))[np.newaxis, :])).float()
    clear_image = torch.from_numpy(np.ascontiguousarray(clear_image.transpose((2, 0, 1))[np.newaxis, :])).float()
    # 归一化
    blur_image = normalize_data(blur_image)
    clear_image = normalize_data(clear_image)

    return blur_image, clear_image


def data_agument(img, mode):
    """
    将数据进行一定随机的转化
    :param img: 图片数据
    :param mode: 变化类型
    :return: 转换后的图片数据
    """
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)  # np.flipud()用于翻转列表，将矩阵进行上下翻转
    elif mode == 2:
        return np.rot90(img)  # #将矩阵img逆时针旋转90°
    elif mode == 3:
        return np.flipud(np.rot90(img))  # #旋转90°后在上下翻转
    elif mode == 4:
        return np.rot90(img, k=2)  # 将矩阵img逆时针旋转90°*k
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        return print("error")
