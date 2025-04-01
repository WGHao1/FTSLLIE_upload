import os
import torch.utils.data as data
import torchvision.transforms as T
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

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
        img = self.preproc(img)

        gt_path = os.path.join(self.target_root, self.gts[idx])
        gt = Image.open(gt_path).convert("RGB")

        gt = self.preproc_gt(gt)

        if self.color_tuning:
            return img, gt, 'a' + self.imgs[idx], 'a' + self.imgs[idx]
        else:
            return img, gt, fn

    def __len__(self):
        return len(self.imgs)
