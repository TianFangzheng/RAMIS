import os
import os.path
import glob
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_paths_and_transform(train_path):
    transform = train_transform
    glob_rgb = os.path.join(train_path)
    def get_gt_paths(p):
        ps = p.split('/')
        pnew = '/'.join(ps[:-3]+['gt'] + ps[-2:])
        return pnew
    paths_rgb = sorted(glob.glob(glob_rgb))
    paths_gt = [get_gt_paths(p) for p in paths_rgb]
    paths = {"rgb": paths_rgb, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_image = np.array(img_file)  # in the range [0,255]
    img_file.close()
    return rgb_image

def gt_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # 判断图片是否为二值图或灰度图
    img_file = img_file.convert('1')
    mode = img_file.mode
    assert mode == '1', "The image is not a binary image "
    gt_image = np.array(img_file)
    img_file.close()
    return gt_image


def train_transform(rgb, target):
    # 将 RGB 和 Ground Truth 合并在一起，组成4通道图像
    rgb = rgb.astype(np.uint8)
    target = target.astype(np.uint8)*255
    img = np.concatenate((rgb, np.expand_dims(target, axis=2)), axis=2)

    # 数据增强
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 转为PIL格式
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),  # 缩放到固定大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor()  # 转换为张量
    ])
    img = transform(img)

    # 分离 RGB 和 Ground Truth
    rgb = img[:3, :, :]
    target = img[3, :, :]

    transform_rgb = transforms.Compose([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)])

    # 颜色变化并且归一化 RGB 图像
    rgb = transform_rgb(rgb)
    target = target.unsqueeze(0)

    return rgb, target


class Train_Data(data.Dataset):
    """A data loader for the Kitti train dataset"""
    def __init__(self, train_path):
        paths, transform = get_paths_and_transform(train_path)
        self.paths = paths
        self.transform = transform

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        target = gt_read(self.paths['gt'][index]) if self.paths['gt'][index] is not None else None
        return rgb, target

    def __getitem__(self, index):
        rgb, target = self.__getraw__(index)
        rgb, target = self.transform(rgb, target)

        candidates = {"rgb":rgb, "gt":target}
        items = {
            key: val
            for key, val in candidates.items() if val is not None}
        return items

    def __len__(self):
        return len(self.paths['rgb'])

if __name__ == '__main__':
    pass