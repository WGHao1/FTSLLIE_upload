import argparse
import os
import kornia
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
import models_vevid
from datasets import LowLightDataset
from tools import saver

def get_args():
    parser = argparse.ArgumentParser('Breaking Downing the Darkness')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')
    parser.add_argument('-m1', '--model1', type=str, default='ILL', help='Model1 Name')
    parser.add_argument('-m2', '--model2', type=str, default='NOI', help='Model2 Name')
    parser.add_argument('-m3', '--model3', type=str, default='COL', help='Model3 Name')
    parser.add_argument('-m4', '--model4', type=str, default='COL', help='Model4 Name')

    parser.add_argument('-m1w', '--model1_weight', type=str, default='./checkpoints/best.pth',
                        help='Combined Model Weight')

    parser.add_argument('--save_extra', default=True, action='store_true', help='save intermediate outputs or not')
    parser.add_argument('--comment', type=str, default='LOL',
                        help='Project comment')
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')

    parser.add_argument('--data_path', type=str, default='./data/LOL/eval',
                        help='the root folder of dataset')
    args = parser.parse_args()
    return args


def load_combined_pth(model1, model2, model3, model4, combined_pth_path):
    combined_state_dict = torch.load(combined_pth_path)

    state_dict_1 = combined_state_dict["model1"]
    state_dict_2 = combined_state_dict["model2"]
    state_dict_3 = combined_state_dict["model3"]
    state_dict_4 = combined_state_dict["model4"]

    model1.load_state_dict(state_dict_1)
    model2.load_state_dict(state_dict_2)
    model3.load_state_dict(state_dict_3)
    model4.load_state_dict(state_dict_4)

class ModelFTSNet(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super().__init__()
        self.eps = 1e-6
        self.model_1net = model1(in_channels=1, out_channels=1)
        self.model_2net = model2(in_channels=2, out_channels=1)
        self.model_3net = model3(in_channels=6, out_channels=2)
        self.model_4net = model4(in_channels=5, out_channels=1)
        load_combined_pth(self.model_1net, self.model_2net, self.model_3net, self.model_4net, opt.model1_weight)

    def noise_syn_exp(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def forward(self, image, image_gt):

        texture_in, cb_in, cr_in = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)
        texture_gt, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image_gt), 1, dim=1)

        texture_in_down = texture_in
        texture_illumi = self.model_1net(texture_in_down)
        illumi = texture_illumi
        texture_illumi = torch.clamp(texture_illumi, 0., 1.)
        texture_ia = texture_in / torch.clamp_min(texture_illumi, self.eps)
        texture_nss = []
        makeNoise_k_strength = []
        noise_k_strength = []

        for strength in [0, illumi, (2 * illumi), (3 * illumi), (4 * illumi)]:
            attention = self.noise_syn_exp(texture_illumi, strength=strength)
            makeNoise_k_strength.append(attention)
            texture_res = self.model_2net(torch.cat([texture_ia, attention], dim=1))
            noise_k_strength.append(texture_res)
            texture_ns = texture_ia + texture_res
            texture_nss.append(texture_ns)

        texture_nss = torch.cat(texture_nss, dim=1).detach()
        texture_fd = self.model_4net(texture_nss)

        image_ia_ycbcr = kornia.color.rgb_to_ycbcr(torch.clamp(image / (texture_illumi + self.eps), 0, 1))
        _, cb_ia, cr_ia = torch.split(image_ia_ycbcr, 1, dim=1)
        colors = self.model_3net(torch.cat([texture_in, cb_in, cr_in, texture_fd, cb_ia, cr_ia], dim=1))

        cb_out, cr_out = torch.split(colors, 1, dim=1)
        cb_out = torch.clamp(cb_out, 0, 1)
        cr_out = torch.clamp(cr_out, 0, 1)

        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_fd, cb_out, cr_out], dim=1))
        image_out = torch.clamp(image_out, 0, 1)

        return image_out

def evaluation(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': False,
                  'num_workers': opt.num_workers}
    val_set = LowLightDataset(opt.data_path)
    val_generator = DataLoader(val_set, **val_params)
    val_generator = tqdm.tqdm(val_generator)

    model1 = getattr(models_vevid, opt.model1)
    model2 = getattr(models_vevid, opt.model2)
    model3 = getattr(models_vevid, opt.model3)
    model4 = getattr(models_vevid, opt.model4)
    model = ModelFTSNet(model1, model2, model3, model4)

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    model.eval()
    for iter, (data, target, name) in enumerate(val_generator):
        with torch.no_grad():
            if opt.num_gpus == 1:
                data = data.cuda()
                target = target.cuda()

            image_out = model(data, target)
            if opt.save_extra:
                saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_image_out')

if __name__ == '__main__':
    opt = get_args()
    evaluation(opt)