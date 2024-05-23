import torch
import torch.nn as nn
import torch.nn.functional as F


class GSA(nn.Module):
    def __init__(self):
        super(GSA, self).__init__()
        # linear_net
        self.liner_layer = nn.Sequential()
        self.liner_layer.add_module('linear1', nn.Linear(in_features=160, out_features=320))
        self.liner_layer.add_module('relu', nn.ReLU())
        self.liner_layer.add_module('linear2', nn.Linear(in_features=320, out_features=160))
        self.liner_layer.add_module('sigmoid', nn.Sigmoid())
        # conv_net1
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1', nn.Conv2d(in_channels=160, out_channels=320, kernel_size=1, stride=1, padding=0, bias=True))
        self.conv1.add_module('conv2', nn.Conv2d(in_channels=320, out_channels=160, kernel_size=1, stride=1, padding=0, bias=True))
        # 融合
        self.conv2 = nn.Conv2d(in_channels=160*2, out_channels=80, kernel_size=1, stride=1, padding=0, bias=True)
        # 融合输出
        self.conv3 = nn.Conv2d(in_channels=80*5, out_channels=400, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input_frame):
        num_frame = len(input_frame)
        now_frame = 2
        Frames = []
        for i in range(num_frame):
            if i is not now_frame:
                frame_temp = torch.cat((input_frame[now_frame], input_frame[i]), dim=1)
                # 通道注意力机制
                linear_temp = F.adaptive_avg_pool2d(frame_temp, (1, 1)).squeeze()
                if len(linear_temp.shape) == 1:
                    linear_temp = linear_temp.unsqueeze(dim=0)
                linear_out = self.liner_layer(linear_temp)
                linear_out = linear_out.reshape(*linear_out.shape, 1, 1)
                # 空间注意力机制
                spatial = Spatial_Attention().cuda()
                spatial_temp = spatial(frame_temp)
                # 卷积降维
                conv_out = self.conv1(frame_temp)
                Frame_temp1 = linear_out * conv_out
                Frame_temp2 = spatial_temp * conv_out
                Frame_temp = torch.cat((Frame_temp1, Frame_temp2), dim=1)
                Frame_temp = self.conv2(Frame_temp)
                Frames.append(Frame_temp)
            else:
                Frames.append(input_frame[now_frame])
        Frame_temp = torch.cat(Frames, dim=1)
        Frame = self.conv3(Frame_temp)
        return Frame


class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=(5-1)//2, dilation=1, groups=1, bias=False)

    def forward(self, input):
        input_avg = torch.mean(input, 1).unsqueeze(1)
        input_max= torch.max(input, 1)[0].unsqueeze(1)
        input_a_m = torch.cat((input_max, input_avg), dim=1)
        output1 = self.conv1(input_a_m)
        sp = torch.sigmoid(output1)
        return sp

