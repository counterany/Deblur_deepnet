import torch
import torch.nn as nn


# RDB_g32/g16...的基本单元，模块化以方便代码编写
class Rdb_g(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Rdb_g, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv.add_module('Relu', nn.ReLU())

    def forward(self, in_put):
        out_put = self.conv(in_put)
        out_put = torch.cat((in_put, out_put), 1)
        return out_put


# RDB基础结构...
class RDB_Cell(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super(RDB_Cell, self).__init__()
        next_in_channels = in_channels
        self.cell = nn.Sequential()
        for i in range(layer_num):
            self.cell.add_module(f'conv{i}', Rdb_g(in_channels=next_in_channels, out_channels=out_channels))
            next_in_channels += out_channels
        self.conv = nn.Conv2d(in_channels=next_in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                              bias=True)

    def forward(self, in_put):
        out_put = self.cell(in_put)
        out_put = self.conv(out_put)
        out_put += in_put
        return out_put


# 多层RDB单元网络
class RDBs(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num, RDBs_layer_num):
        super(RDBs, self).__init__()
        self.RDBs_layer_num = RDBs_layer_num
        # 不用sequential，用ModuleList可方便获取层数
        self.rdb_net = nn.ModuleList()
        for i in range(self.RDBs_layer_num):
            self.rdb_net.append(RDB_Cell(in_channel, out_channel, layer_num))
        self.conv1 = nn.Conv2d(in_channels=in_channel*self.RDBs_layer_num, out_channels=in_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, in_put):
        out_put = []
        for i in range(self.RDBs_layer_num):
            in_put = self.rdb_net[i](in_put)
            out_put.append(in_put)
        out_put = torch.cat(out_put, 1)
        out_put = self.conv1(out_put)
        out_put = self.conv2(out_put)
        return out_put


# 整个RDB_RNN网络
class RDB_RNN_Net(nn.Module):
    def __init__(self):
        super(RDB_RNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.RDBCell1 = nn.Sequential()
        self.RDBCell1.add_module('rdb', RDB_Cell(in_channels=16, out_channels=16, layer_num=3))
        self.RDBCell1.add_module('conv', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2,padding=2, bias=True))
        self.RDBCell2 = nn.Sequential()
        self.RDBCell2.add_module('rdb', RDB_Cell(in_channels=32, out_channels=24, layer_num=3))
        self.RDBCell2.add_module('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True))
        # RDBs层
        self.Rdbs = RDBs(in_channel=80, out_channel=32, layer_num=3, RDBs_layer_num=15)
        # 隐藏状态层
        self.hidden_layer = nn.Sequential()
        self.hidden_layer.add_module('conv1', nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True))
        self.hidden_layer.add_module('rdb', RDB_Cell(in_channels=16, out_channels=16, layer_num=3))
        self.hidden_layer.add_module('conv2', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, now_frame, hidden_data):
        # now_frame = now_frame.cpu()  ####
        out_put = self.conv1(now_frame)
        out_put = self.RDBCell1(out_put)
        out_put = self.RDBCell2(out_put)
        out_put = torch.cat((out_put, hidden_data), dim=1)
        out_put = self.Rdbs(out_put)
        hidden_frame = self.hidden_layer(out_put)
        return out_put, hidden_frame
