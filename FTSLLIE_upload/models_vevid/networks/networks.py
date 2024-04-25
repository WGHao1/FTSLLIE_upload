from models_vevid.networks.modules import *
import numpy as np
from models_vevid.networks.base_layers import *
from torch.fft import fft2, fftshift, ifft2

class BaseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(BaseNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

        self.up1 = Up(256, 64, bilinear=True, norm=norm)
        self.up2 = Up(128, 32, bilinear=True, norm=norm)
        self.up3 = Up(64, 32, bilinear=True, norm=norm)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# class IAN(BaseNet):
#     def __init__(self, in_channels=1, out_channels=1, norm=True):
#         super(IAN, self).__init__(in_channels, out_channels, norm)

# ------------------------------------------------------------vevid亮度调剂网络
class CSDN_Tem(nn.Module):
    # DEC-Net网络重新设计，将卷积网络替换成深度可分离卷积，
    # 每个深度可分离卷积层由1个核大小为 3×3，步幅为 1 的depthwise convolution
    # 和 核大小为 1×1、步长为 1 的pointwise convolution组成。
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_point_conv = nn.Sequential(
            # 深度卷积和点卷积
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1 )
        )

    def forward(self, input):
        out = self.depth_point_conv(input)
        return out

# only point-wise:逐点卷积

class Hist_adjust(nn.Module):
    # 好像没用到
    def __init__(self, in_ch, out_ch):
        super(Hist_adjust, self).__init__()
        self.point_conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0,groups=1)
    def forward(self, input):
        out = self.point_conv(input)
        return out

#Hornet：递归门控卷积

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution：标准卷积
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # SiLU()：Sigmoid Linear Unit
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    """
    每个样本的下降路径（随机深度）（应用于残差块的主路径时）。
    这与我为EfficientNet等网络创建的DropConnect
    impl相同，然而，
    原始名称具有误导性，因为“Drop
    Connect”是另一篇论文中不同形式的辍学。。。
    参见讨论：https: // github.com / tensorflow / tpu / issues / 494  # issuecomment-532968956。。。我选择了
    将层和参数名称更改为“drop - path”，而不是将DropConnect混合作为层名称并使用
    以“存活率”为论据。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class HorLayerNorm(nn.Module):
    # 递归门控卷积部分：#https://ar5iv.labs.arxiv.org/html/2207.14284
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).# https://ar5iv.labs.arxiv.org/html/2207.14284
    r“”“LayerNorm，支持两种数据格式：channels_last（默认）或channels_first。
    输入中尺寸的顺序。channels_last对应的输入
    形状（batch_size、高度、宽度、通道），而channels_first对应于输入
    带形状（batch_size、channels、height、width）。#https://ar5iv.labs.arxiv.org/html/2207.14284
    “”“
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # by iscyy/air
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape gnconv [512]by iscyy/air
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)

        return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class HorBlock(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = HorLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape # [512]
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class C3HB(nn.Module):
    # CSP HorBlock with 3 convolutions by iscyy/yoloair
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(HorBlock(c_) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

def normalize(inputs, mode ,a ,b):
    assert mode in ["standardize", "normalize"]
    if mode == "normalize":
        # data nomalization into [0, 1]
        max_val, min_val = inputs.max(), inputs.min()
        inputs_nor = (inputs - min_val) / (max_val - min_val)
        inputs = a + (inputs_nor * (b-a))
        # inputs = inputs / (inputs.max() - inputs.min()).clamp(min=1e-8)
    else:
        # data standardize
        mean, var = inputs.mean(), inputs.var()
        inputs = (inputs - mean) / var
    return inputs


class IAN(nn.Module):
    '''https://li-chongyi.github.io/Proj_Zero-DCE.html'''

    def __init__(self, in_channels=1, out_channels=1, norm=True, n=8, device='cuda'):#n=8
        '''
        Args
        --------
          n: number of iterations of LE(x) = LE(x) + alpha * LE(x) * (1-LE(x)).
          return_results: [4, 8] => return the 4-th and 8-th iteration results.
        '''
        super().__init__()

        self.concat_input = Concat()
        # Parameter
        self.Gauss = torch.as_tensor(
                        np.array([[0.0947416, 0.118318, 0.0947416],
                                [ 0.118318, 0.147761, 0.118318],
                                [0.0947416, 0.118318, 0.0947416]]).astype(np.float32)
                        )
        self.Gauss_kernel = self.Gauss.expand(1, 1, 3, 3).to(device)
        self.w = nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device).data.fill_(0.72)
        self.sigma = nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device).data.fill_(2.0)

        self.n = n
        number_f = 32

        #C3HB Horblock
        self.conv1 = CSDN_Tem(1, number_f)#self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = C3HB(number_f, number_f)
        self.conv3 = C3HB(number_f, number_f)
        self.conv4 = C3HB(number_f, number_f)
        self.conv5 = CSDN_Tem(number_f * 2, number_f)#self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.conv6 = CSDN_Tem(number_f * 2, number_f)#self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.conv7 = CSDN_Tem(number_f * 2, 2 * n)#self.conv7 = nn.Conv2d(64, 2 * n, kernel_size=3, padding=1, bias=True)#
        self.conv8 = CSDN_Tem(number_f * 2, 3 * n)#elf.conv8 = nn.Conv2d(64, 3 * n, kernel_size=3, padding=1, bias=True)#

        # 全是深度可分离卷积()
        self.Tem_conv1 = CSDN_Tem(1, number_f)#self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.Tem_conv2 = CSDN_Tem(number_f, number_f)
        self.Tem_conv3 = CSDN_Tem(number_f, number_f)
        self.Tem_conv4 = CSDN_Tem(number_f, number_f)
        self.Tem_conv5 = CSDN_Tem(number_f * 2, number_f)#self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.Tem_conv6 = CSDN_Tem(number_f * 2, number_f)#self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.Tem_conv7 = CSDN_Tem(number_f * 2, 2 * n)#self.conv7 = nn.Conv2d(64, 2 * n, kernel_size=3, padding=1, bias=True)#
        self.Tem_conv8 = CSDN_Tem(number_f * 2, 3 * n)#elf.conv8 = nn.Conv2d(64, 3 * n, kernel_size=3, padding=1, bias=True)#

        # 逐点卷积
        self.b_conv1 = Hist_adjust(1, number_f)
        self.b_conv2 = Hist_adjust(number_f, number_f)
        self.b_conv3 = Hist_adjust(number_f, number_f)
        self.b_conv4 = Hist_adjust(number_f, number_f)
        self.b_conv5 = Hist_adjust(number_f, 2)
        self.b_conv6 = Hist_adjust(number_f, 3)


        # 全是深度可分离卷积()
        self.tem_conv1 = CSDN_Tem(1, number_f)#self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.tem_conv2 = CSDN_Tem(number_f, number_f)
        self.tem_conv3 = CSDN_Tem(number_f, number_f)
        self.tem_conv4 = CSDN_Tem(number_f, number_f)
        self.tem_conv5 = CSDN_Tem(number_f, 2)#self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.tem_conv6 = CSDN_Tem(number_f, 3)#self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#


        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def standard_illum_map(self, I, ratio=1, blur=False):
        self.w.clamp_(0.01, 0.99)
        self.sigma.clamp_(0.1, 10)

        I = torch.log(I + 1.)
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        I_std = torch.std(I, dim=[2, 3], keepdim=True)
        I_min = I_mean - self.sigma * I_std
        I_max = I_mean + self.sigma * I_std
        I_range = I_max - I_min
        I_out = torch.clamp((I - I_min) / I_range, min=0.0, max=1.0)
        # Transfer to gamma correction, center intensity is w
        I_out = I_out ** (-1.442695 * torch.log(self.w))
        return I_out

    def set_parameter(self, w=None):
        if w is None:
            self.w.requires_grad = True
        else:
            self.w.data.fill_(w)
            self.w.requires_grad = False

    def get_parameter(self):
        if self.w.device.type == 'cuda':
            w = self.w.detach().cpu().numpy()
            sigma = self.sigma.detach().cpu().numpy()
        else:
            w = self.w.cpu().numpy()
            sigma = self.sigma.cpu().numpy()
        return w, sigma

    # 第一种vevid方法
    def forward(self, input_I, lite=False):
        # 这是U-Net形式的vevid图像增强
        with torch.no_grad():

            esp = 1e-8
        input_L = input_I
        vevid_input = input_L#hsv_img

        zero_idx = torch.where(vevid_input == 0)

        vevid_input[zero_idx] = esp
        # 这一大段是为了避免分母为0

        # 这里的网络结构为：
        # **   --------   **
        #   **  ------  **
        #     ** ---- **
        #         **
        out1 = self.gelu(self.Tem_conv1(vevid_input))
        out2 = self.Tem_conv2(out1)
        out3 = self.Tem_conv3(out2)
        out4 = self.Tem_conv4(out3)
        out5 = self.gelu(self.Tem_conv5(torch.cat((out4, out3), 1)))
        out6 = self.gelu(self.Tem_conv6(torch.cat((out5, out2), 1)))


        if lite:
            alpha_stacked = self.lrelu(self.Tem_conv7(torch.cat((out6, out1), 1)))
            alphas = torch.split(alpha_stacked, 2, 1)
            for i in range(self.n):
                #al = torch.unsqueeze(alphas[i][:, 0, :, :], dim=1)
                b = normalize(torch.unsqueeze(alphas[i][:, 0, :, :], dim=1), 'normalize', 0.3, 1)
                G = normalize(torch.unsqueeze(alphas[i][:, 1, :, :], dim=1), 'normalize', 0.3, 1)
                vevid_phase = torch.atan2(-G * (vevid_input + b), vevid_input)
                vevid_phase_norm = (vevid_phase - vevid_phase.min()) / (vevid_phase.max() - vevid_phase.min())
                out = vevid_phase_norm
        else:
            alpha_stacked = self.lrelu(self.Tem_conv8(torch.cat((out6, out1), 1)))
            alphas = torch.split(alpha_stacked, 3, 1)
            for i in range(self.n):

                #T = torch.unsqueeze(alphas[i][:, 0, :, :], dim=1)
                S = normalize(torch.unsqueeze(alphas[i][:, 0, :, :], dim=1), 'normalize', 0.3, 1)
                b = normalize(torch.unsqueeze(alphas[i][:, 1, :, :], dim=1), 'normalize', 0.3, 1)
                G = normalize(torch.unsqueeze(alphas[i][:, 2, :, :], dim=1), 'normalize', 0.3, 1)

                vevid_kernel = S
                vevid_input_f = fft2(vevid_input+b)
                img_vevid = ifft2(vevid_input_f * fftshift(torch.exp(-1j * vevid_kernel)))
                vevid_phase = torch.atan2(G * torch.imag(img_vevid), vevid_input)

                vevid_phase_norm = (vevid_phase - vevid_phase.min()) / (vevid_phase.max() - vevid_phase.min())
                out = vevid_phase_norm
        return out


class IAN_S(nn.Module):
    '''https://li-chongyi.github.io/Proj_Zero-DCE.html'''

    def __init__(self, in_channels=1, out_channels=1, norm=True, n=8, device='cuda'):  # n=8
        '''
        Args
        --------
          n: number of iterations of LE(x) = LE(x) + alpha * LE(x) * (1-LE(x)).
          return_results: [4, 8] => return the 4-th and 8-th iteration results.
        '''
        super().__init__()

        self.concat_input = Concat()
        # Parameter
        self.Gauss = torch.as_tensor(
            np.array([[0.0947416, 0.118318, 0.0947416],
                      [0.118318, 0.147761, 0.118318],
                      [0.0947416, 0.118318, 0.0947416]]).astype(np.float32)
        )
        self.Gauss_kernel = self.Gauss.expand(1, 1, 3, 3).to(device)
        self.w = nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device).data.fill_(0.72)
        self.sigma = nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device).data.fill_(2.0)

        self.n = n
        number_f = 32

        # C3HB Horblock
        self.conv1 = CSDN_Tem(1, number_f)  # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = C3HB(number_f, number_f)
        self.conv3 = C3HB(number_f, number_f)
        self.conv4 = C3HB(number_f, number_f)
        self.conv5 = CSDN_Tem(number_f * 2,
                              number_f)  # self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.conv6 = CSDN_Tem(number_f * 2,
                              number_f)  # self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.conv7 = CSDN_Tem(number_f * 2,
                              2 * n)  # self.conv7 = nn.Conv2d(64, 2 * n, kernel_size=3, padding=1, bias=True)#
        self.conv8 = CSDN_Tem(number_f * 2,
                              3 * n)  # elf.conv8 = nn.Conv2d(64, 3 * n, kernel_size=3, padding=1, bias=True)#

        # 全是深度可分离卷积()
        self.Tem_conv1 = CSDN_Tem(1, number_f)  # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.Tem_conv2 = CSDN_Tem(number_f, number_f)
        self.Tem_conv3 = CSDN_Tem(number_f, number_f)
        self.Tem_conv4 = CSDN_Tem(number_f, number_f)
        self.Tem_conv5 = CSDN_Tem(number_f * 2,
                                  number_f)  # self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.Tem_conv6 = CSDN_Tem(number_f * 2,
                                  number_f)  # self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.Tem_conv7 = CSDN_Tem(number_f * 2,
                                  1)  # self.conv7 = nn.Conv2d(64, 2 * n, kernel_size=3, padding=1, bias=True)#
        self.Tem_conv8 = CSDN_Tem(number_f * 2,
                                  3 * n)  # elf.conv8 = nn.Conv2d(64, 3 * n, kernel_size=3, padding=1, bias=True)#

        # 逐点卷积
        self.b_conv1 = Hist_adjust(1, number_f)
        self.b_conv2 = Hist_adjust(number_f, number_f)
        self.b_conv3 = Hist_adjust(number_f, number_f)
        self.b_conv4 = Hist_adjust(number_f, number_f)
        self.b_conv5 = Hist_adjust(number_f, 2)
        self.b_conv6 = Hist_adjust(number_f, 3)

        # 全是深度可分离卷积()
        self.tem_conv1 = CSDN_Tem(1, number_f)  # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.tem_conv2 = CSDN_Tem(number_f, number_f)
        self.tem_conv3 = CSDN_Tem(number_f, number_f)
        self.tem_conv4 = CSDN_Tem(number_f, number_f)
        self.tem_conv5 = CSDN_Tem(number_f, 2)  # self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#
        self.tem_conv6 = CSDN_Tem(number_f, 3)  # self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)#

        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def standard_illum_map(self, I, ratio=1, blur=False):
        self.w.clamp_(0.01, 0.99)
        self.sigma.clamp_(0.1, 10)

        I = torch.log(I + 1.)
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        I_std = torch.std(I, dim=[2, 3], keepdim=True)
        I_min = I_mean - self.sigma * I_std
        I_max = I_mean + self.sigma * I_std
        I_range = I_max - I_min
        I_out = torch.clamp((I - I_min) / I_range, min=0.0, max=1.0)
        # Transfer to gamma correction, center intensity is w
        I_out = I_out ** (-1.442695 * torch.log(self.w))
        return I_out

    def set_parameter(self, w=None):
        if w is None:
            self.w.requires_grad = True
        else:
            self.w.data.fill_(w)
            self.w.requires_grad = False

    def get_parameter(self):
        if self.w.device.type == 'cuda':
            w = self.w.detach().cpu().numpy()
            sigma = self.sigma.detach().cpu().numpy()
        else:
            w = self.w.cpu().numpy()
            sigma = self.sigma.cpu().numpy()
        return w, sigma

    # 第一种vevid方法
    def forward(self, input_I, lite=False):
        # 这是U-Net形式的vevid图像增强
        with torch.no_grad():

            esp = 1e-8
        input_L = input_I
        vevid_input = input_L  # hsv_img

        zero_idx = torch.where(vevid_input == 0)

        vevid_input[zero_idx] = esp
        # 这一大段是为了避免分母为0

        # 这里的网络结构为：
        # **   --------   **
        #   **  ------  **
        #     ** ---- **
        #         **
        out1 = self.gelu(self.Tem_conv1(vevid_input))
        out2 = self.Tem_conv2(out1)
        out3 = self.Tem_conv3(out2)
        out4 = self.Tem_conv4(out3)
        out5 = self.gelu(self.Tem_conv5(torch.cat((out4, out3), 1)))
        out6 = self.gelu(self.Tem_conv6(torch.cat((out5, out2), 1)))

        out = self.lrelu(self.Tem_conv7(torch.cat((out6, out1), 1)))

        return out



class ANSN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(ANSN, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

        self.up1 = Up(256, 64, bilinear=True, norm=norm)
        self.up2 = Up(128, 32, bilinear=True, norm=norm)
        self.up3 = Up(64, 32, bilinear=True, norm=norm)
        self.outc = OutConv(32, out_channels, act=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class FuseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=False):
        super(FuseNet, self).__init__()
        self.inc = AttentiveDoubleConv(in_channels, 32, norm=norm, leaky=False)
        self.down1 = AttentiveDown(32, 64, norm=norm, leaky=False)
        self.down2 = AttentiveDown(64, 64, norm=norm, leaky=False)
        self.up1 = AttentiveUp(128, 32, bilinear=True, norm=norm, leaky=False)
        self.up2 = AttentiveUp(64, 32, bilinear=True, norm=norm, leaky=False)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits






if __name__ == '__main__':
    for key in FuseNet(4, 2).state_dict().keys():
        print(key)
