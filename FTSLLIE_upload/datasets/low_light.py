import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

# 如果不想花时间把数据集中的破损图片找出来后删除掉，只需要设置LOAD_TRUNCATED_IMAGES = True，
# 在程序开头只需要加上两行代码就行了
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# 这样在遇到截断的JPEG时，程序就会跳过去，读取另一张图片了。
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2 as cv

import torch
from PIL import Image
from torchvision import transforms

class LowLightFDataset(data.Dataset):
    def  __init__(self, root, image_split='images', targets_split='targets', training=True):
        self.root = root
        self.num_instances = 8
        self.img_root = os.path.join(root, image_split)
        self.target_root = os.path.join(root, targets_split)
        self.training = training
        print('----加载的图像文件夹:', root, image_split, targets_split, '----')
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))
        # sorted() 函数对所有可迭代的对象进行排序操作。
        # os.listdir()： 列出路径下所有的文件
        # 上面调试没问题

        # names = [img_name.split('_')[0] + '.' + img_name.split('.')[-1] for img_name in self.imgs] # 注释掉的原因是这里把图像的后缀重复了一遍，如：102.png.png
        names = [img_name.split('_')[0] for img_name in self.imgs]
        # 上面调试没问题
        # self.imgs = list(filter(lambda img_name: img_name.split('_')[0] + '.' + img_name.split('.')[-1] in self.gts, self.imgs)) # 注释掉的原因是这里建立不了图像列表

        self.imgs = list(
            filter(lambda img_name: img_name.split('_')[0] in self.gts, self.imgs))

        # filter函数是数组里的一个方法，它主要起到的是过滤作用，filter()创建一个新的数组，新数组中的元素是通过检查指定数组中符合条件的所有元素。
        # filter函数使用的地方非常的广泛简单举一个例子：      # 数组去重操作：对数组array中所有相同的元素进行去重复操作

        # Lambda 函数又称匿名函数，即用句子实现函数的功能
        # lambda 函数实例：
        #     add = lambda x, y: x + y
        #     add(1, 2)
        # 结果：        3

        self.gts = list(filter(lambda gt: gt in names, self.gts))

        print('文件', image_split, '和', targets_split, '的图像数量', len(self.imgs), len(self.gts)) # 显示图像数量

        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')
        imgs = []
        for i in range(self.num_instances):
            # img_path = os.path.join(self.img_root, f"{fn}_{i}.{ext}") # 注释掉的原因是没有图像是这样命名的“{fn}_{i}.{ext}”
            img_path = os.path.join(self.img_root, f"{fn}.{ext}")
            imgs += [self.preproc(Image.open(img_path).convert("RGB"))]

        # print(f"{fn}.{ext}")

        if self.training:
            random.shuffle(imgs)
        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        # print(img_path, gt_path)
        return torch.stack(imgs, dim=0), gt, fn

    def __len__(self):
        return len(self.gts)


class LowLightFDataset_edges1(data.Dataset):
    def __init__(self, root, image_split='images', targets_split='targets', image_edges_split='images_edges1', targets_edges_split='targets_edges1', training=True):
        self.root = root
        self.num_instances = 8
        self.img_root = os.path.join(root, image_split)
        self.target_root = os.path.join(root, targets_split)
        self.img_edges_root = os.path.join(root, targets_split)
        self.target_edges_root = os.path.join(root, targets_split)
        self.training = training
        print('----加载的图像文件夹:', root, image_split, targets_split, image_edges_split, targets_edges_split,'----')
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))
        self.imgs_edges = list(sorted(os.listdir(self.img_edges_root)))
        self.gts_edges = list(sorted(os.listdir(self.target_edges_root)))
        # sorted() 函数对所有可迭代的对象进行排序操作。
        # os.listdir()： 列出路径下所有的文件
        # 上面调试没问题

        # names = [img_name.split('_')[0] + '.' + img_name.split('.')[-1] for img_name in self.imgs] # 注释掉的原因是这里把图像的后缀重复了一遍，如：102.png.png
        names = [img_name.split('_')[0] for img_name in self.imgs]
        # 上面调试没问题
        # self.imgs = list(filter(lambda img_name: img_name.split('_')[0] + '.' + img_name.split('.')[-1] in self.gts, self.imgs)) # 注释掉的原因是这里建立不了图像列表

        self.imgs = list(
            filter(lambda img_name: img_name.split('_')[0] in self.gts, self.imgs))
        # filter函数是数组里的一个方法，它主要起到的是过滤作用，filter()创建一个新的数组，新数组中的元素是通过检查指定数组中符合条件的所有元素。
        # filter函数使用的地方非常的广泛简单举一个例子：      # 数组去重操作：对数组array中所有相同的元素进行去重复操作

        # Lambda 函数又称匿名函数，即用句子实现函数的功能
        # lambda 函数实例：
        #     add = lambda x, y: x + y
        #     add(1, 2)
        # 结果：        3
        self.gts = list(filter(lambda gt: gt in names, self.gts))

        self.imgs_edges = list(
            filter(lambda img_name: img_name.split('_')[0] in self.gts_edges, self.imgs_edges))
        self.gts_edges = list(filter(lambda gts_edges: gts_edges in names, self.gts_edges))

        print('文件', image_split, '和', targets_split, '和', image_edges_split, '和', targets_edges_split, '的图像数量', len(self.imgs), len(self.gts), len(self.imgs_edges), len(self.gts_edges)) # 显示图像数量

        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_edges = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_edges_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')
        imgs = []
        imgs_edges = []
        for i in range(self.num_instances):
            img_path = os.path.join(self.img_root, f"{fn}.{ext}")
            imgs += [self.preproc(Image.open(img_path).convert("RGB"))]

        for i in range(self.num_instances):
            img_edges_path = os.path.join(self.img_edges_root, f"{fn}.{ext}")
            imgs_edges += [self.preproc(Image.open(img_edges_path).convert("RGB"))]
        # print(f"{fn}.{ext}")

        # if self.training:
        #     random.shuffle(imgs)
        #     random.shuffle(imgs_edges)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt_edges_path = os.path.join(self.target_edges_root, self.gts[idx])

        gt = Image.open(gt_path).convert("RGB")
        gt_edges = Image.open(gt_edges_path).convert("RGB")

        gt = self.preproc_gt(gt)
        gt_edges = self.preproc_edges_gt(gt_edges)

        # print(img_path, gt_path)
        return torch.stack(imgs, dim=0), gt, torch.stack(imgs_edges, dim=0), gt_edges, fn

    def __len__(self):
        return len(self.gts)

class LowLightFDataset_edges2(data.Dataset):
    def __init__(self, root, image_split='images', targets_split='targets', image_edges_split='images_edges2', targets_edges_split='targets_edges2', training=True):
        self.root = root
        self.num_instances = 8
        self.img_root = os.path.join(root, image_split)
        self.target_root = os.path.join(root, targets_split)
        self.img_edges_root = os.path.join(root, targets_split)
        self.target_edges_root = os.path.join(root, targets_split)
        self.training = training
        print('----加载的图像文件夹:', root, image_split, targets_split, image_edges_split, targets_edges_split,'----')
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))
        self.imgs_edges = list(sorted(os.listdir(self.img_edges_root)))
        self.gts_edges = list(sorted(os.listdir(self.target_edges_root)))
        # sorted() 函数对所有可迭代的对象进行排序操作。
        # os.listdir()： 列出路径下所有的文件
        # 上面调试没问题

        # names = [img_name.split('_')[0] + '.' + img_name.split('.')[-1] for img_name in self.imgs] # 注释掉的原因是这里把图像的后缀重复了一遍，如：102.png.png
        names = [img_name.split('_')[0] for img_name in self.imgs]
        # 上面调试没问题
        # self.imgs = list(filter(lambda img_name: img_name.split('_')[0] + '.' + img_name.split('.')[-1] in self.gts, self.imgs)) # 注释掉的原因是这里建立不了图像列表

        self.imgs = list(
            filter(lambda img_name: img_name.split('_')[0] in self.gts, self.imgs))
        # filter函数是数组里的一个方法，它主要起到的是过滤作用，filter()创建一个新的数组，新数组中的元素是通过检查指定数组中符合条件的所有元素。
        # filter函数使用的地方非常的广泛简单举一个例子：      # 数组去重操作：对数组array中所有相同的元素进行去重复操作

        # Lambda 函数又称匿名函数，即用句子实现函数的功能
        # lambda 函数实例：
        #     add = lambda x, y: x + y
        #     add(1, 2)
        # 结果：        3
        self.gts = list(filter(lambda gt: gt in names, self.gts))

        self.imgs_edges = list(
            filter(lambda img_name: img_name.split('_')[0] in self.gts_edges, self.imgs_edges))
        self.gts_edges = list(filter(lambda gts_edges: gts_edges in names, self.gts_edges))

        print('文件', image_split, '和', targets_split, '和', image_edges_split, '和', targets_edges_split, '的图像数量', len(self.imgs), len(self.gts), len(self.imgs_edges), len(self.gts_edges)) # 显示图像数量

        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_edges = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_edges_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')
        imgs = []
        imgs_edges = []
        for i in range(self.num_instances):
            img_path = os.path.join(self.img_root, f"{fn}.{ext}")
            imgs += [self.preproc(Image.open(img_path).convert("RGB"))]

        for i in range(self.num_instances):
            img_edges_path = os.path.join(self.img_edges_root, f"{fn}.{ext}")
            imgs_edges += [self.preproc(Image.open(img_edges_path).convert("RGB"))]
        # print(f"{fn}.{ext}")

        # if self.training:
        #     random.shuffle(imgs)
        #     random.shuffle(imgs_edges)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt_edges_path = os.path.join(self.target_edges_root, self.gts[idx])

        gt = Image.open(gt_path).convert("RGB")
        gt_edges = Image.open(gt_edges_path).convert("RGB")

        gt = self.preproc_gt(gt)
        gt_edges = self.preproc_edges_gt(gt_edges)

        # print(img_path, gt_path)
        return torch.stack(imgs, dim=0), gt, torch.stack(imgs_edges, dim=0), gt_edges, fn

    def __len__(self):
        return len(self.gts)

class LowLightFDatasetEval(data.Dataset):
    def __init__(self, root, targets_split='targets', training=True):
        self.root = root
        self.num_instances = 1
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.training = training

        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')
        imgs = []
        for i in range(self.num_instances):
            img_path = os.path.join(self.img_root, f"{fn}.{ext}")
            imgs += [self.preproc(Image.open(img_path).convert("RGB"))]

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        # print(img_path, gt_path)
        return torch.stack(imgs, dim=0), gt, fn

    def __len__(self):
        return len(self.gts)

class LowLightDataset_size(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.color_tuning = color_tuning

        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')

        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # 定义目标大小
        target_size = (640, 512)
        # 创建resize变换对象
        resizer = transforms.Resize(target_size)
        # 应用resize变换到PIL图像上
        img = resizer(img)
        # img = cv.resize(img, [640, 512], interpolation=cv.INTER_CUBIC)  # 把图像尺寸统一

        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        # 定义目标大小
        target_size = (640, 512)
        # 创建resize变换对象
        resizer = transforms.Resize(target_size)
        # 应用resize变换到PIL图像上
        gt = resizer(gt)
        # gt = cv.resize(gt, [640, 512], interpolation=cv.INTER_CUBIC)  # 把图像尺寸统一

        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return img, gt, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, fn

    def __len__(self):
        return len(self.imgs)

class LowLightDataset_LOL_Real(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.color_tuning = color_tuning


        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.imgs, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.gts, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')

        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # img = cv.resize(img, [640, 512], interpolation=cv.INTER_CUBIC)  # 把图像尺寸统一

        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        # gt = cv.resize(gt, [640, 512], interpolation=cv.INTER_CUBIC)  # 把图像尺寸统一
        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return img, gt, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, fn

    def __len__(self):
        return len(self.imgs)

class LowLightDataset(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.color_tuning = color_tuning


        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')

        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # img = cv.resize(img, [640, 512], interpolation=cv.INTER_CUBIC)  # 把图像尺寸统一

        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        # gt = cv.resize(gt, [640, 512], interpolation=cv.INTER_CUBIC)  # 把图像尺寸统一
        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return img, gt, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, fn

    def __len__(self):
        return len(self.imgs)

class LowLightDataset_edges1(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.img_edges_root = os.path.join(root, 'images_edges1')
        self.target_edges_root = os.path.join(root, 'targets_edges1')
        self.color_tuning = color_tuning
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))
        self.imgs_edges = list(sorted(os.listdir(self.img_edges_root)))
        self.gts_edges = list(sorted(os.listdir(self.target_edges_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))
        self.imgs_edges = list(filter(lambda img_edges_name: img_edges_name in self.gts_edges, self.imgs_edges))
        self.gts_edges = list(filter(lambda gt_edges: gt_edges in self.imgs_edges, self.gts_edges))

        print(len(self.imgs), len(self.gts))
        print(len(self.imgs_edges), len(self.gts_edges))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_edges = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt_edges = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')

        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        img_edges_path = os.path.join(self.img_edges_root, self.imgs[idx])
        img_edges = Image.open(img_edges_path).convert("RGB")
        img_edges = self.preproc(img_edges)

        gt_edges_path = os.path.join(self.target_edges_root, self.gts[idx])
        gt_edges = Image.open(gt_edges_path).convert("RGB")
        gt_edges = self.preproc_gt(gt_edges)

        if self.color_tuning:
            return img, gt, img_edges, gt_edges, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, img_edges, gt_edges, fn

    def __len__(self):
        return len(self.imgs)

class LowLightDataset_edges2(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.img_edges_root = os.path.join(root, 'images_edges2')
        self.target_edges_root = os.path.join(root, 'targets_edges2')
        self.color_tuning = color_tuning
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))
        self.imgs_edges = list(sorted(os.listdir(self.img_edges_root)))
        self.gts_edges = list(sorted(os.listdir(self.target_edges_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))
        self.imgs_edges = list(filter(lambda img_edges_name: img_edges_name in self.gts_edges, self.imgs_edges))
        self.gts_edges = list(filter(lambda gt_edges: gt_edges in self.imgs_edges, self.gts_edges))

        print(len(self.imgs), len(self.gts))
        print(len(self.imgs_edges), len(self.gts_edges))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_edges = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt_edges = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        fn, ext = self.gts[idx].split('.')

        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        img_edges_path = os.path.join(self.img_edges_root, self.imgs[idx])
        img_edges = Image.open(img_edges_path).convert("RGB")
        img_edges = self.preproc(img_edges)

        gt_edges_path = os.path.join(self.target_edges_root, self.gts[idx])
        gt_edges = Image.open(gt_edges_path).convert("RGB")
        gt_edges = self.preproc_gt(gt_edges)

        if self.color_tuning:
            return img, gt, img_edges, gt_edges, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, img_edges, gt_edges, fn

    def __len__(self):
        return len(self.imgs)

class LowLightDatasetReverse(data.Dataset):
    def __init__(self, root, targets_split='targets', color_tuning=False):
        self.root = root
        self.img_root = os.path.join(root, 'images')
        self.target_root = os.path.join(root, targets_split)
        self.color_tuning = color_tuning
        self.imgs = list(sorted(os.listdir(self.img_root)))
        self.gts = list(sorted(os.listdir(self.target_root)))

        self.imgs = list(filter(lambda img_name: img_name in self.gts, self.imgs))
        self.gts = list(filter(lambda gt: gt in self.imgs, self.gts))

        print(len(self.imgs), len(self.gts))
        self.preproc = T.Compose(
            [T.ToTensor()]
        )
        self.preproc_gt = T.Compose(
            [T.ToTensor()]
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")
        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return gt, img, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            fn, ext = os.path.splitext(self.imgs[idx])
            return gt, img, '%03d' % int(fn) + ext

    def __len__(self):
        return len(self.imgs)
