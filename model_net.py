from random import random

from net_model.RDB_RNN import RDB_RNN_Net
from net_model.GSA import GSA
import torch.nn as nn
import torch
# from torchsummary import summary


class ESTRNN(nn.Module):
    def __init__(self):
        super(ESTRNN, self).__init__()
        # RDB_RNN
        self.RDB_Net = RDB_RNN_Net()
        # 全局时空注意力机制
        self.GSA = GSA()
        # 重构
        self.re_constructor = nn.Sequential()
        self.re_constructor.add_module('deconv1', nn.ConvTranspose2d(in_channels=400, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.re_constructor.add_module('deconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.re_constructor.add_module('conv', nn.Conv2d(16, 3, 5, 1, 2, bias=True))

    def forward(self, in_put):
        out_puts = []
        all_frames = []
        # 获得数据中的特征（批次大小， 帧数，通道数，高度，宽度）
        batch_size, frame_num, channel_num, height, width = in_put.shape
        # 将长高缩短到网络可接受的固定尺度
        base_height = int(height/4)
        base_width = int(width/4)
        # 生成一个零值帧以解决边界帧没有上一帧的问题
        hidden_frame = torch.zeros(batch_size, 16, base_height, base_width).cuda()
        for i in range(frame_num):
            now_frame, hidden_frame = self.RDB_Net(in_put[:, i, :, :, :], hidden_frame)
            all_frames.append(now_frame)
        # all_frames = torch.tensor([item.cpu().detach().numpy() for item in all_frames])
        for i in range(2, frame_num - 2):  # 4个结果
            # 递归输入
            out_put = self.GSA(all_frames[i-2:i+3])  # (400,
            out_put = self.re_constructor(out_put)
            out_puts.append(out_put.unsqueeze(dim=1))
        out_put = torch.cat(out_puts, dim=1)
        return out_put


# model = ESTRNN()
# summary(model, (8, 3, 256, 256))
