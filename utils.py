import os
import cv2
import numpy as np
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from os.path import dirname, join


# 优化器
class Optimizer:
    def __init__(self, model, learning_rate, learning_rate_strategy='cosine'):
        self.optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        if learning_rate_strategy == 'multi_step':
            self.scheduler = lr_scheduler.MultiStepLR(self.optim, [200, 400], gamma=0.5)  # 动态调整
        elif learning_rate_strategy == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, T_max=500, eta_min=1e-8)  # 遇险退货策略
        elif learning_rate_strategy == 'cosineW':
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=2, eta_min=1e-8)

    def step(self):
        self.optim.step()

    def get_lr(self):
        lr = self.optim.param_groups[0]['lr']
        return torch.tensor(lr)

    def zero_grad(self):
        self.optim.zero_grad()

    def learn_rate_schedule(self):
        self.scheduler.step()


# 损失函数
def grad_l(x, y):
    gray = rgb_gray()
    x_gray = gray(x)
    y_gray = gray(y)
    x_ax_loss = torch.pow(gradient(x_gray, "x") - gradient(y_gray, 'x'), 2)
    y_ax_loss = torch.pow(gradient(x_gray, "y") - gradient(y_gray, 'y'), 2)
    g_l = torch.mean(x_ax_loss + y_ax_loss)
    return g_l


class my_loss(nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, net_output, truth_frames, sc=1, gc=1, mc=1):
        # batch_size, frame_number, channel_number, height, width = truth_frames.shape
        # truth_frames = truth_frames.reshape(batch_size, frame_number * channel_number, height, width)
        if isinstance(net_output, (list, tuple)):
            batch_size, frame_number, channel_number, height, width = truth_frames.shape
            _truth = list()
            _truth.append(truth_frames)
            truth_frames = truth_frames.reshape(batch_size, frame_number*channel_number, height, width)
            _truth.append(
                F.interpolate(truth_frames, size=(height // 2, width // 2), mode='bilinear', align_corners=False).reshape(
                    batch_size, frame_number, channel_number, height // 2, width // 2))
            _truth.append(
                F.interpolate(truth_frames, size=(height // 4, width // 4), mode='bilinear', align_corners=False).reshape(
                    batch_size, frame_number, channel_number, height // 4, width // 4))
            assert len(net_output) == len(_truth)
            length = len(net_output)
            loss_logger = []
            for i in range(length):
                temp_loss = self.loss_signal(net_output[i], _truth[i])
                loss_logger.append(temp_loss)
        else:
            loss_logger = self.loss_signal(net_output, truth_frames.cuda(), sc, gc, mc)
        return loss_logger

    def loss_signal(self, net_output, truth_frames, sc=1, gc=4, mc=8):
        if len(net_output.shape) == 5:
            batch_size, frame_number, channel_number, height, width = net_output.shape
            net_output = net_output.reshape(batch_size * frame_number, channel_number, height, width)
            truth_frames = truth_frames.reshape(batch_size * frame_number, channel_number, height, width)
        loss_logger = {}
        # 论文中的损失函数(L1范数
        # net_output = net_output.cpu()
        L1_loss = self.L1_char_loss_mean(net_output, truth_frames)
        loss_logger['L1_loss'] = L1_loss * mc
        # ssim损失值
        ssim_loss = ssim_l(net_output, truth_frames)
        loss_logger['SSIM'] = ssim_loss * sc
        # 光滑相似
        grad_loss = grad_l(net_output, truth_frames)
        loss_logger['Grad'] = grad_loss * gc
        loss_all = L1_loss + ssim_loss + grad_loss
        loss_logger['All'] = loss_all
        return loss_logger


    def L1_char_loss_mean(self, x, y):
        eps = torch.tensor(1e-3)
        temp = torch.add(x, -y)
        temp_sq = temp * temp
        # 取平均
        temp_sq_mean = torch.mean(temp_sq, 1, True)
        # 取根号
        loss = torch.sqrt(temp_sq_mean + eps * eps)
        loss = torch.mean(loss)
        return loss


class rgb_gray(nn.Module):
    def __init__(self):
        super(rgb_gray, self).__init__()
        kernel = [0.299, 0.587, 0.114]
        self.weight = torch.tensor(kernel).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        gray = F.conv2d(x, self.weight)
        return gray


def gradient(input_tensor, direction):
    """
    计算梯度并平滑
    :param input_tensor: 计算梯度的数据
    :param direction: 梯度的方向
    :return: 返回梯度
    """
    b = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], 1).cuda()
    input_tensor = torch.cat((input_tensor, b), 3)
    a = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 1, input_tensor.shape[3]).cuda()
    input_tensor = torch.cat((input_tensor, a), 2)
    c = [[0, 0], [-1, 1]]
    c = torch.FloatTensor(c).cuda()
    # 下边设置平滑核，分别为x方向和y方向
    smooth_kernel_x = torch.reshape(c, (1, 1, 2, 2))
    smooth_kernel_y = smooth_kernel_x.permute([0, 1, 3, 2])  # 改变数据的维度，这里将3和2互换了
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    weight = nn.Parameter(data=kernel, requires_grad=False).cuda()  # 将不可以训练的数据转化为可以训练的数据
    gradient_orig = torch.abs(F.conv2d(input_tensor, weight, stride=1, padding=0))  # 将输入和权重卷积
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))  # torch.div，点除操作
    return grad_norm


# 度量SSIM
def gauss(window_size, sigma):
    # 高斯窗
    x_data, y_data = np.mgrid[-window_size // 2 + 1:window_size // 2 + 1, -window_size // 2 + 1:window_size // 2 + 1]
    x_data = torch.Tensor(x_data).unsqueeze(0).unsqueeze(0).cuda()
    y_data = torch.Tensor(y_data).unsqueeze(0).unsqueeze(0).cuda()
    g = torch.exp(-((x_data ** 2 + y_data ** 2) / (2.0 * sigma ** 2))).cuda()
    return g / torch.sum(g)


def ssim(x, y, mean_metric=True, size=11, sigma=1.5):
    # 配置参数
    window = gauss(size, sigma)
    k1 = torch.Tensor([0.01]).cuda()  # 亮度相似系数
    k2 = torch.Tensor([0.03]).cuda()  # 对比度相似系数
    L = torch.Tensor([1]).cuda() # 图像像素的范围（0~1）
    c1 = torch.pow(k1 * L, 2).cuda()
    c2 = torch.pow(k2 * L, 2).cuda()
    weight = nn.Parameter(data=window, requires_grad=False).cuda()
    # 亮度相似
    mean1 = F.conv2d(x, weight)
    mean2 = F.conv2d(y, weight)
    E1_sq = mean1 * mean1
    E2_sq = mean2 * mean2
    E12 = mean1 * mean2
    l_loss = (2 * E12 + c1) / (E1_sq + E2_sq + c1)
    # 对比度相似c,和结构相似s
    relate11 = F.conv2d(x * x, weight) - E1_sq
    relate22 = F.conv2d(y * y, weight) - E2_sq
    relate12 = F.conv2d(x * y, weight) - E12
    s_c_loss = (2 * relate12 + c2)/(relate11 + relate22 + c2)
    # ssim
    ssim_loss = l_loss * s_c_loss
    if mean_metric is True:
        ssim_loss = torch.mean(ssim_loss)
    return ssim_loss


def ssim_l(x, y):
    if len(x.shape) == 5:
        batch_size, frame_number, channel_number, height, width = x.shape
        x = x.reshape(batch_size * frame_number, channel_number, height, width)
        y = y.reshape(batch_size * frame_number, channel_number, height, width)
    # 分通道计算
    x_1 = x[:, 0:1, :, :]
    y_1 = y[:, 0:1, :, :]
    sl1 = ssim(x_1, y_1)
    x_2 = x[:, 1:2, :, :]
    y_2 = y[:, 1:2, :, :]
    sl2 = ssim(x_2, y_2)
    x_3 = x[:, 2:3, :, :]
    y_3 = y[:, 2:3, :, :]
    sl3 = ssim(x_3, y_3)
    sl = (sl1+sl2+sl3) / 3.0
    loss_sl = 1 - sl
    return loss_sl


def ssim1(x, y, mean_metric=True, size=11, sigma=1.5):
    # 配置参数
    window = gauss(size, sigma).cpu()
    k1 = torch.Tensor([0.01])  # 亮度相似系数
    k2 = torch.Tensor([0.03])  # 对比度相似系数
    L = torch.Tensor([1]) # 图像像素的范围（0~1）
    c1 = torch.pow(k1 * L, 2)
    c2 = torch.pow(k2 * L, 2)
    weight = nn.Parameter(data=window, requires_grad=False)
    # 亮度相似
    mean1 = F.conv2d(x, weight)
    mean2 = F.conv2d(y, weight)
    E1_sq = mean1 * mean1
    E2_sq = mean2 * mean2
    E12 = mean1 * mean2
    l_loss = (2 * E12 + c1) / (E1_sq + E2_sq + c1)
    # 对比度相似c,和结构相似s
    relate11 = F.conv2d(x * x, weight) - E1_sq
    relate22 = F.conv2d(y * y, weight) - E2_sq
    relate12 = F.conv2d(x * y, weight) - E12
    s_c_loss = (2 * relate12 + c2)/(relate11 + relate22 + c2)
    # ssim
    ssim_loss = l_loss * s_c_loss
    if mean_metric is True:
        ssim_loss = torch.mean(ssim_loss)
    return ssim_loss


def ssim_l1(x, y):
    x = torch.tensor(x).permute((2,0,1))
    y = torch.tensor(y).permute((2,0,1))
    # 分通道计算
    x_1 = x[0:1, :, :]
    y_1 = y[0:1, :, :]
    sl1 = ssim1(x_1, y_1)
    x_2 = x[1:2, :, :]
    y_2 = y[1:2, :, :]
    sl2 = ssim1(x_2, y_2)
    x_3 = x[2:3, :, :]
    y_3 = y[2:3, :, :]
    sl3 = ssim1(x_3, y_3)
    sl = (sl1+sl2+sl3) / 3.0
    return sl


# 度量PSNR
def MSE(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    __, ___, h, w = x.shape
    x = x.cpu()
    y = y.cpu()
    mse = 0
    for key in range(h):
        for mark in range(w):
            temp = x[:, :, key, mark] - y[:, :, key, mark]
            temp = torch.pow(temp, 2)
            mse += temp
    return torch.mean(mse)


def PSNR(I, K, max_value=1.0):
    I = I.clamp(0, max_value).round().cpu()
    K = K.clamp(0, max_value).round().cpu()
    Mse = 0.0
    if len(I.shape) == 5:
        __, ___, channel, high, width = I.shape
        for i in range(channel):
            Mse_temp = MSE(I[:, :, i, :, :], K[:, :, i, :, :])
            Mse += Mse_temp
        temp = max_value * max_value / Mse
        psnr = 10 * torch.log10(temp)
    return psnr


def MSE1(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    h, w = x.shape
    mse = 0
    for key in range(h):
        for mark in range(w):
            temp = x[key, mark] - y[ key, mark]
            temp = temp * temp
            mse += temp
    return mse/(h*w)

def PSNR1(I, K):
    # I = I.clamp(0, 1.0).round()
    # K = K.clamp(0, 1.0).round()
    Mse = 0.0
    psnr_all = 0.0
    for i in range(3):
        Mse_temp = MSE1(I[:, :, i], K[:, :, i])
        Mse += Mse_temp
        Mse = Mse/3
        temp = 1.0 * 1.0 / Mse
        psnr = 10 * np.log10(temp)
        psnr_all += psnr
    return psnr_all/3
# 记录器
class RecordLogger:
    def __init__(self, save_path, model, dataset):
        file_path = join(save_path, model + '_' + dataset, 'log.txt')
        self.save_path = dirname(file_path)
        self.check_dir(file_path)
        self.logger = open(file_path, 'a+')
        self.dict = {}

    def check_dir(self, path):
        dir = dirname(path)
        os.makedirs(dir, exist_ok=True)  # 创建目录

    def __call__(self, *args, prefix=''):
        info = prefix
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        self.logger.flush()  # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区

    # 记录值
    def record(self, name, epoch, value):
        if name in self.dict:
            self.dict[name][epoch] = value
        else:
            self.dict[name] = {}
            self.dict[name][epoch] = value


# 保存图像
def save_final_images(file_path, result2):
    """
    保存增强后的图片
    :param file_path: 保存的文件地址
    :param result2: 增强后的图片矩阵
    :return: 无
    """
    result2 = result2.cpu().detach().numpy()  # 将结果转化为numpy类型，用于后续保存图片
    result2 = np.squeeze(result2)
    cv2.imwrite(file_path, result2 * 255.0)


# video_img("G:\\undergraduate\\dataset\\BSD_2ms16ms\\test")
'''
# image path
im_dir = "G:\\undergraduate\\dataset\\BSD_2ms16ms\\test\\015\\Blur\\RGB"
# output video path
save_video_dir = 'G:\\undergraduate\\save_video'
if not os.path.exists(save_video_dir):
    os.makedirs(save_video_dir)
# set saved fps
fps = 10
# get frames list
frames = sorted(os.listdir(im_dir))
# w,h of image
img = cv2.imread(os.path.join(im_dir, frames[0]))
img_size = (img.shape[1], img.shape[0])
# get seq name
seq_name = os.path.dirname(im_dir).split('/')[-1]
# splice video_dir
video_dir = os.path.join(save_video_dir, seq_name + '.avi')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames:
    f_path = os.path.join(im_dir, frame)
    image = cv2.imread(f_path)
    videowriter.write(image)
    print(frame + " has been written!")

videowriter.release()
'''
