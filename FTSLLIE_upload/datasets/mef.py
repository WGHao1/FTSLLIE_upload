import os
import random
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from natsort import os_sorted
# -*- coding: UTF-8 -*-
import os
import imageio
import shutil


class MEFDataset(data.Dataset):
    def __init__(self, root):

        self.img_root = root

        self.numbers = list(sorted(os.listdir(self.img_root)))
        # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        # sorted 是内置函数，可以对任何可迭代的对象进行排序，并返回一个排序后的列表。它不会修改原对象，而是返回一个新的列表。
        # list 列表
        print(len(self.numbers))

        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        #     transforms.compose 是 PyTorch 中的一个函数，它可以将多个图像变换组合在一起。
        #     它接受一个变换的序列作为输入，并返回一个新的变换，当这个变换被应用到图像时，
        #     它会依次应用每一个组成变换。这可以使代码更加简洁，并且可以确保所有变换是按照指定的顺序应用。

    def __getitem__(self, idx):
        # 类中__getitem__的作用
        # 当一个python类中定义了__getitem__函数，则其实例对象能够通过下标来进行索引数据。
        number = self.numbers[idx]
        im_dir = os.path.join(self.img_root, number)
        # os.path.join是用来合并路径的函数。它可以把多个路径组合在一起，并正确处理斜杠和反斜杠。
        fn1, fn2 = tuple(random.sample(os.listdir(im_dir), k=2))
        # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        # random.sample() 方法可以随机地从指定列表中提取出N个不同的元素，列表的维数没有限制。
        fp1 = os.path.join(im_dir, fn1)
        fp2 = os.path.join(im_dir, fn2)
        img1 = Image.open(fp1).convert("RGB")
        img2 = Image.open(fp2).convert("RGB")
        img1 = self.preproc(img1)
        img2 = self.preproc(img2)

        fn1 = f'{number}_{fn1}'
        fn2 = f'{number}_{fn2}'
        return img1, img2, fn1, fn2

    def __len__(self):
        return len(self.numbers)
