import argparse
import datetime
import os
import traceback
import multiprocessing as mp
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import models_vevid
from datasets import LowLightDataset, LowLightFDataset
from models_vevid import PSNR, SSIM, CosineLR
from tools import SingleSummaryWriter
from tools import saver, mutils

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def get_args():
    parser = argparse.ArgumentParser('Breaking Downing the Darkness')
    parser.add_argument('--num_gpus', type=int, default=2, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=2, help='The number of images per batch among all devices')
    parser.add_argument('-m', '--model', type=str, default='IAN',
                        help='Model Name')
    parser.add_argument('--comment', type=str, default='ILL_checkpoints',
                        help='Project comment')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--scratch', action='store_true')

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--no_sche', action='store_true')

    parser.add_argument('--optim', type=str, default='adam', help='select optimizer for training, '
                                                                  'suggest using \'admaw\' until the'
                                                                  ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--data_path', type=str, default='./data/LOL',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--saved_path', type=str, default='logs/')
    args = parser.parse_args()
    return args


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class ModelINet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.restor_loss = models_vevid.MSELoss()
        self.wtv_loss = models_vevid.WTVLoss2()
        self.ssiml1loss = models_vevid.SSIML1Loss(channels=1)
        self.ssim_loss = models_vevid.SSIMLoss(channels=1)
        self.l2_loss = models_vevid.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.model = model(in_channels=1, out_channels=1)
        self.eps = 1e-2

    def forward(self, image, image_gt, training=True):
        # def forward(self, image, image_gt, training=False):
        if training:
            image = image.squeeze(0)
            image_gt = image_gt.repeat(8, 1, 1, 1)
        else:
            image = image
            image_gt = image_gt

        texture_in, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)
        texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image_gt), 1, dim=1)

        texture_in_down = texture_in
        texture_gt_down = texture_gt
        illumi = self.model(texture_in_down)
        texture_out = texture_in_down / torch.clamp_min(illumi, self.eps)
        l2_ssim_loss = self.l2_loss(texture_out, texture_gt_down) + 1.2 * self.ssim_loss(texture_out, texture_gt_down)

        return texture_out, illumi, l2_ssim_loss


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    timestamp = mutils.get_formatted_time()
    opt.saved_path = opt.saved_path + f'/{opt.comment}/{timestamp}'
    opt.log_path = opt.log_path + f'/{opt.comment}/{timestamp}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'num_workers': opt.num_workers}

    training_set = LowLightFDataset(os.path.join(opt.data_path, 'train'), image_split='images',
                                    targets_split='targets')
    training_generator = DataLoader(training_set, **training_params)

    val_set = LowLightDataset(os.path.join(opt.data_path, 'eval'), targets_split='targets')
    val_generator = DataLoader(val_set, **val_params)

    model = getattr(models_vevid, opt.model)

    model = ModelINet(model)

    writer = SingleSummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = CosineLR(optimizer, opt.lr, opt.num_epochs)
    epoch = 0
    step = 0
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, (data, target, name) in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:

                    data, target = data.cuda(), target.cuda()

                    texture_out, texture_attention, l2_ssim_loss = model(data, target, training=True)
                    loss = l2_ssim_loss

                    optimizer.zero_grad()
                    loss.sum().backward()
                    optimizer.step()


                    step += 1

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            if opt.no_sche:
                scheduler.step()

            saver.base_url = os.path.join(opt.saved_path, 'results', '%03d' % epoch)

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_ls = []
                psnrs = []
                ssims = []

                for iter, (data, target, name) in enumerate(val_generator):
                    with torch.no_grad():

                        data, target = data.cuda(), target.cuda()

                        texture_in, _, _ = torch.split(kornia.color.rgb_to_ycbcr(data), 1, dim=1)
                        texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(target), 1, dim=1)

                        texture_out, texture_attention, l2_ssim_loss = model.module(data, target, training=False)

                        loss = l2_ssim_loss
                        loss_ls.append(loss.item())

                        psnr = PSNR(texture_out, texture_gt)
                        ssim = SSIM(texture_out, texture_gt).item()
                        psnrs.append(psnr)
                        ssims.append(ssim)

                loss = np.mean(np.array(loss_ls))
                psnr = np.mean(np.array(psnrs))
                ssim = np.mean(np.array(ssims))

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
                        epoch, opt.num_epochs, loss, psnr, ssim))

                loss = format(loss, '.5f')
                psnr = format(psnr, '.5f')
                ssim = format(ssim, '.5f')

                save_checkpoint(model,
                                f'{"loss"}_{loss}_{opt.model}_{"%03d" % epoch}_{"psnr"}_{psnr}_{"ssim"}_{ssim}.pth')

                model.train()

    except KeyboardInterrupt:
        save_checkpoint(model, f'{opt.model}_{epoch}_keyboardInterrupt.pth')
        writer.close()
    writer.close()

def save_checkpoint(model, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))

if __name__ == '__main__':

    print(f"num of CPU: {mp.cpu_count()}")
    opt = get_args()
    train(opt)
