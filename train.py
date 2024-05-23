import os
import random
from os.path import join
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from test import file_path_main
from net_model.model_net import ESTRNN
import torch
import torch.nn as nn
from utils import Optimizer, my_loss, PSNR, ssim_l, RecordLogger
from deal_dataset import DeblurDataset
from torch.utils.data import DataLoader


def train(learning_rate, learning_rate_strategy, epoch, path, data_format, frames, batch_size, ):
    # 配置随机种子，保证每次训练随机数一致
    seed = 39
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # 实例化网络
    model = ESTRNN()
    model = nn.DataParallel(model).cuda()
    # 定义优化算法
    opt = Optimizer(model, learning_rate, learning_rate_strategy=learning_rate_strategy)
    # 定义损失函数
    e_loss = my_loss()
    # 实例化数据类
    path_train = join(path, 'train')
    dataset = DeblurDataset(path_train, data_format, frames)
    train_load = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    path_valid = join(path, 'valid')
    dataset1 = DeblurDataset(path_valid, data_format, frames)
    valid_load = DataLoader(dataset=dataset1, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    # logger
    logger = RecordLogger(".\\record", 'ESTRNN', 'BSD')
    logger.writer = SummaryWriter(logger.save_path)
    logger('model structure', model)
    metrics_logger = {}
    loss_logger = {}
    loss_all = []
    lr = {}
    for e in range(epoch):
        print('epoch:', e)
        model.train()
        loss_all.append(0.0)
        loss_key = ['SSIM', 'L1_loss', 'Grad', 'All']
        metrics_key = ['SSIM', 'PSNR']
        for key, data in enumerate(train_load):
            in_put, truth_image = data
            truth_image1 = truth_image[:, 2:6, :, :, :]
            out_put = model(in_put)
            loss = e_loss(out_put, truth_image1)
            loss_all[e] += loss['All'].detach().item()
            for k in loss_key:
                loss_logger[f"{e}_{k}"] = loss[k].detach().item()
                logger.writer.add_scalar(k + '_loss+train', loss_logger[f"{e}_{k}"], e)
            print(f"e:{e}, key:{key}, SSIM:{loss['SSIM']}; L1_loss:{loss['L1_loss']}; Grad:{loss['Grad']}; loss:{loss['All']}; loss_ALL:{loss_all[e]}")
            for k in metrics_key:
                if k == 'PSNR':
                    metrics_logger[f"{e}_{k}"] = PSNR(truth_image1, out_put)
                    logger.writer.add_scalar('metrics_PSNR_train', metrics_logger[f"{e}_PSNR"], e)
                elif k == 'SSIM':
                    metrics_logger[f"{e}_{k}"] = loss['SSIM']
                    logger.writer.add_scalar('metrics_SSIM_train', metrics_logger[f"{e}_SSIM"], e)
            opt.zero_grad()
            loss['All'].backward()
            # 剪切梯度(解决梯度爆炸的问题）
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            # 更新权重
            opt.step()
            # 记录
            logger.record('loss_train', e, loss_logger[f"{e}_All"])
            logger.record('metrics_PSNR_train', e, metrics_logger[f"{e}_PSNR"])
            logger.record('metrics_SSIM_train', e, metrics_logger[f"{e}_SSIM"])
            lr['e'] = opt.get_lr().detach().item()
            logger.writer.add_scalar('lr', lr['e'], e)
        for key in loss_key:
            print(f"{key}:{loss[key]};")
        logger(logger.dict)
        loss_min = np.min(loss_all)
        valid(valid_load=valid_load, model=model, loss=e_loss, logger=logger, e=e)
        if loss_all[e] == loss_min:
            torch.save({'state_dict': model.state_dict(), 'epoch': e, 'optimizer': opt.optim.state_dict()},
                       'MyNet_' + '_best.pkl')
        if e == 0 or e % 10 == 0:
            torch.save({'state_dict': model.state_dict(), 'epoch': e, 'optimizer': opt.optim.state_dict()},
                       'MyNet_' + str(e) + '_best.pkl')
            valid(valid_load=valid_load, model=model, loss=e_loss, logger=logger, e=e)
        if e == 0 or e % 100 == 0:
            file_path_main(e)
    print('已经保存最优的模型参数')


def valid(valid_load, model, loss, logger, e):
    model.eval()
    loss_key = ['SSIM', 'L1_loss', 'Grad', 'All']
    metrics_key = ['SSIM', 'PSNR']
    metrics_logger = {}
    loss_logger = {}
    with torch.no_grad():
        for key, data in enumerate(valid_load):
            in_put, truth_image = data
            out_put = model(in_put)
            truth_image1 = truth_image[:, 2:6, :, :, :]
            losses = loss(out_put, truth_image1)
            if isinstance(out_put, (list, tuple)):
                out_put = out_put[0]
            for k in loss_key:
                loss_logger[f"{k}_valid"] = losses[k].detach().item()
            for k in metrics_key:
                if k == 'PSNR':
                    metrics_logger[f"{k}_valid"] = PSNR(truth_image1, out_put)
                elif k == 'SSIM':
                    metrics_logger[f"{k}_valid"] = losses['SSIM']
                    # 记录
            logger.record('loss_valid', e, loss_logger["All_valid"])
            logger.record('metrics_PSNR_valid', e, metrics_logger["PSNR_valid"])
            logger.record('metrics_SSIM_valid', e, metrics_logger["SSIM_valid"])


if __name__ == "__main__":
    epoch = 500
    b_s = 8
    p_s = [256, 256]
    l_r = 5e-4
    l_r_s = 'cosine'
    path = ".\\dataset\\BSD_2ms16ms"
    d_f = 'RGB'
    frame = 8
    train(learning_rate=l_r, learning_rate_strategy=l_r_s, epoch=epoch, path=path, data_format=d_f, frames=frame,
          batch_size=b_s)
