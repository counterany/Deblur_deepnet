from os.path import join, dirname
import torch
from collections import OrderedDict
from skimage.metrics import structural_similarity as SSIM_1
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter
from net_model.model_net import ESTRNN
import os
import cv2
import numpy as np
from utils import PSNR1, ssim_l1, RecordLogger
from utils import PSNR1, ssim_l1


def load_images(img):
    """
    归一化单张图片的矩阵
    :param img: 图片绝对地址
    :return: 图片矩阵数值
    """
    img = np.array(img, dtype='float32') / 255.0  # 将image格式转化为numpy格式并归一化到0~1之间（转化为灰度图片）
    img_norm = np.float32((img - np.min(img))
                          / np.maximum((np.max(img) - np.min(img)), 0.001))  # np.maximum逐元素比较两个array的大小
    return img_norm


def read_load_image2(file_path):
    """
    获得所有图片的矩阵和形状
    :param file_path: 文件夹位置
    :return: 图片名称的集合
    """
    all_img = []
    shape_m = []
    for filename in os.listdir(file_path):  # 返回指定的文件夹包含的文件或文件夹的名字的列表。
        img = cv2.imread(file_path + "/" + filename)
        shape_m.append(img.shape)
        # img = cv2.resize(img, (1024, 1024))
        img = load_images(img)
        all_img.append(img)
    return all_img, shape_m


# 保存为视频
def video_img(path, size, seq, frame_start, frame_end, marks, fps=10):
    file_path = join(path, f'{seq}.avi')
    os.makedirs(dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for i in range(frame_start, frame_end):
        imgs = []
        for j in range(len(marks)):
            img_path = join(path, f"{seq}", '{:06d}.png'.format(i))
            img = cv2.imread(img_path)
            img = cv2.putText(img, marks[j], (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            imgs.append(img)
        frame = np.concatenate(imgs, axis=1)
        video.write(frame)
    video.release()
# video_img(path="F:\\dataset\\test\\GOPR0396_11_00", size=(1280, 720), seq="blur", frame_start=3, frame_end=98, marks=["Blur"], fps=8)
# print("my")
def Normalize(x, centralize=False, normalize=False, val_range=255.0):
    # 正规化数值
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range
    return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x


def file_path_main(C):
    logger = RecordLogger(".\\record", 'test', 'BSD')
    logger.writer = SummaryWriter(logger.save_path)
    # 加载模型参数
    # CKPT_PATH = f".\\MyNet_{e-1}_best.pkl"
    # CKPT_PATH = "MyNet_104_best.pkl"
    CKPT_PATH = C
    # 加载分解网络模型
    checkpoints = torch.load(CKPT_PATH)
    decom_checkpoint = checkpoints['state_dict']
    e = checkpoints['epoch']
    new_decom = OrderedDict()
    for k, v in decom_checkpoint.items():
        name_1 = k[7:]  # remove `module.`
        new_decom[name_1] = v
    model = ESTRNN().cuda()
    model.load_state_dict(new_decom)
    model.eval()

    # H, W = 480, 640
    # val_range = 2.0 ** 8 - 1
    # dataset_path = 'G:\\undergraduate\\dataset\\BSD_2ms16ms\\test'
    dataset_path = "G:\\undergraduate11\\save_video\\epoch_225"
    # dataset_path = 'F:\\dataset\\test\\'
    save_dir_path = f'.\\save_video\\next_new_epoch_{e}'
    seqs = sorted(os.listdir(dataset_path))

    # seq_length = 100
    # results_register = set()
    # psnr = {}
    # ssim = {}
    for seq in seqs:
        # dir_name = join('BSD', "net_model", str(e), 'test')
        img_seq = sorted(os.listdir(f"G:\\undergraduate11\\save_video\\epoch_225\\{seq}\\deblur"))
        seq_length = len(img_seq)
        save_dir = join(save_dir_path, seq)
        os.makedirs(save_dir, exist_ok=True)
        # blur_save_img_path = join(save_dir, "Blur")
        # os.makedirs(blur_save_img_path, exist_ok=True)
        # gt_save_img_path = join(save_dir, "GT")
        # os.makedirs(gt_save_img_path, exist_ok=True)
        deblur_save_img_path = join(save_dir, "deblur")
        os.makedirs(deblur_save_img_path, exist_ok=True)
        suffix = 'png'
        start = 0
        test_frames = 20
        end = test_frames
        while True:
            input_seq = []
            # label_seq = []
            for frame_idx in range(start, end):
                blur_img_path = join(dataset_path, seq, 'deblur', '{}'.format(img_seq[frame_idx]))
                # blur_img_path = join(dataset_path, seq, 'blur', '{:06d}.{}'.format(frame_idx, suffix))
                # sharp_img_path = join(dataset_path, seq, 'Sharp', 'RGB', '{:08d}.{}'.format(frame_idx, suffix))
                # sharp_img_path = join(dataset_path, seq, 'sharp', '{:06d}.{}'.format(frame_idx, suffix))
                blur_img = cv2.imread(blur_img_path).transpose(2, 0, 1)[np.newaxis, ...]
                # blur_img = torch.tensor(blur_img)
                # gt_img = cv2.imread(sharp_img_path)
                # gt_img = gt_img.transpose(2,0,1)[np.newaxis, ...]
                # gt_img = torch.tensor(gt_img)
                input_seq.append(blur_img)
                # label_seq.append(gt_img)
            input_seq = torch.tensor(input_seq).permute([1, 0, 2, 3, 4])
            # label_seq = torch.tensor(label_seq).permute([1, 0, 2, 3, 4])
            # input_seq = torch.cat(input_seq, dim=1).unsqueeze(dim=1)
            # input_seq = np.concatenate(input_seq)[np.newaxis, :]
            model.eval()
            with torch.no_grad():
                # input_seq = Normalize(input_seq.float().cuda(), centralize=True,
                #                    normalize=True, val_range=val_range)
                input_seq = torch.tensor(load_images(input_seq.float())).cuda()
                output_seq = model(input_seq)
                if isinstance(output_seq, (list, tuple)):
                    output_seq = output_seq[0]
                output_seq = output_seq.squeeze(dim=0)
            for frame_idx in range(2, end - start - 2):
                #  blur_img = input_seq.squeeze(dim=0)[frame_idx]
                # blur_img = normalize_reverse(blur_img, centralize=True, normalize=True,
                #                              val_range=val_range)
                # blur_img = blur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                # blur_img = blur_img.astype(np.uint8)
                # blur_img = Image.fromarray(np.clip(blur_img * 255.0, 0, 255.0).astype('uint8'))
                # blur_img = np.clip(blur_img * 255.0, 0, 255.0).astype('uint8')
                # blur_img_path_one = join(blur_save_img_path, '{:08d}_input.{}'.format(frame_idx + start, suffix))
                # gt_img = label_seq[frame_idx]
                # gt_img_path_one = join(gt_save_img_path, '{:08d}_gt.{}'.format(frame_idx + start, suffix))
                deblur_img = output_seq[frame_idx - 2].cpu()
                # deblur_img = normalize_reverse(deblur_img, centralize=True, normalize=True,
                #                                val_range=val_range)
                deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                # psnr[f"{e}_{seq}_{frame_idx + start}_PSNR"] = PSNR1(deblur_img, load_images(gt_img))
                # ssim[f"{e}_{seq}_{frame_idx + start}_SSIM"] = ssim_l1(deblur_img, load_images(gt_img))
                # deblur_img = np.clip(deblur_img, 0, val_range)
                # deblur_img = deblur_img.astype(np.uint8)
                # deblur_img = Image.fromarray(np.clip(deblur_img * 255.0, 0, 255.0).astype('uint8'))
                deblur_img = np.clip(deblur_img * 255.0, 0, 255.0).astype('uint8')
                deblur_img_path_one = join(deblur_save_img_path, '{:08d}_{}.{}'.format(frame_idx + start, 'net_model', suffix))
                # deblur_img.save(deblur_img_path, 'png')
                # blur_img.save(blur_img_path, 'png')
                # cv2.imwrite(gt_img_path_one, gt_img)
                # cv2.imwrite(blur_img_path_one, blur_img)
                # cv2.imwrite(gt_img_path, gt_img)
                cv2.imwrite(deblur_img_path_one, deblur_img)

            if end == seq_length:
                break
            else:
                start = end - 2 - 2
                end = start + test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - test_frames
        '''
        logger('seq {} video result generating ...'.format(seq))
        marks = ['Input', "model", 'GT']
        path = dirname(save_dir)
        frame_start = 2
        frame_end = seq_length - 2
        video_img(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                          marks=marks, fps=15)
        '''
        # logger(f'Test images {e} PSNR : {len(psnr)}', prefix='\n')
        # logger(f'Test {e} PSNR : {np.mean(psnr)}')
        # logger(f'Test {e} SSIM : {np.mean(ssim)}')
        # return psnr, ssim


def file_path_BSD(C):
    logger = RecordLogger(".\\record", 'test', 'BSD')
    logger.writer = SummaryWriter(logger.save_path)
    # 加载模型参数
    CKPT_PATH = C
    # 加载分解网络模型
    checkpoints = torch.load(CKPT_PATH)
    decom_checkpoint = checkpoints['state_dict']
    e = checkpoints['epoch']
    new_decom = OrderedDict()
    for k, v in decom_checkpoint.items():
        name_1 = k[7:]  # remove `module.`
        new_decom[name_1] = v
    model = ESTRNN().cuda()
    model.load_state_dict(new_decom)
    model.eval()
    dataset_path = 'G:\\undergraduate\\dataset\\BSD_2ms16ms\\test'
    save_dir_path = f'.\\save_video\\epoch_{e}'
    seqs = sorted(os.listdir(dataset_path))
    seq_length = 150
    for seq in seqs:
        save_dir = join(save_dir_path, seq)
        os.makedirs(save_dir, exist_ok=True)
        deblur_save_img_path = join(save_dir, "deblur")
        os.makedirs(deblur_save_img_path, exist_ok=True)
        suffix = 'png'
        start = 0
        test_frames = 20
        end = test_frames
        while True:
            input_seq = []
            for frame_idx in range(start, end):
                blur_img_path = join(dataset_path, seq, 'Blur', 'RGB', '{:08d}.{}'.format(frame_idx, suffix))
                blur_img = cv2.imread(blur_img_path).transpose(2, 0, 1)[np.newaxis, ...]
                input_seq.append(blur_img)
            input_seq = torch.tensor(input_seq).permute([1, 0, 2, 3, 4])
            model.eval()
            with torch.no_grad():
                input_seq = torch.tensor(load_images(input_seq.float())).cuda()
                output_seq = model(input_seq)
                if isinstance(output_seq, (list, tuple)):
                    output_seq = output_seq[0]
                output_seq = output_seq.squeeze(dim=0)
            for frame_idx in range(2, end - start - 2):
                deblur_img = output_seq[frame_idx - 2].cpu()
                deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                deblur_img = np.clip(deblur_img * 255.0, 0, 255.0).astype('uint8')
                deblur_img_path_one = join(deblur_save_img_path, '{:08d}_{}.{}'.format(frame_idx + start, 'net_model', suffix))
                cv2.imwrite(deblur_img_path_one, deblur_img)

            if end == seq_length:
                break
            else:
                start = end - 4
                end = start + test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - test_frames


def metrics(my_img1):
        name = ["015", "020", "028", "044", "052", "053", "069", "072", "073", "075","078","080","099","107","109","119","120","121","128","129"]
        # he_img = "G:\\ESTRNN\\ESTRNN-master\\result\\2ms16ms\\015\\DeBlur"
        # my_image_file_path = "./my_image"  # 增强后图像保存的地址
        # he_img, img_shape = read_load_image2(he_img)
        '''
        test_he_image = he_img[2]  # 这里我选择了第1张照片
        test_blur_image = he_img[3]
        '''
        psnr_cal = PSNR_cal()
        ALL_PSNR = []
        ALL_SSIM = []
        for n in name:
            # my_img = f"save_video/other_epoch_164/{n}/deblur"  {n}/deblur
            my_img = join(my_img1, f"{n}/deblur")
            my_img, my_img_shape = read_load_image2(my_img)
            # blur_img = "save_video/other_epoch_112/015/Blur"
            # blur_img, blur_img_shape = read_load_image2(blur_img)
            sharp_image = f"G:\\undergraduate11\\epoch_329/{n}/GT"
            sharp_image, _ = read_load_image2(sharp_image)
            '''
            test_my_image = my_img[1]
            '''
            if len(my_img_shape) == len(my_img_shape):
                my_PSNR = 0.0
                MY_SSIM = 0.0
                for i in range(len(my_img_shape)):
                    test_my_image = my_img[i]
                    test_sharp_image = sharp_image[i]
                    PSNR = psnr_cal(test_my_image, test_sharp_image)
                    # SSIM = ssim_l1(test_my_image, test_sharp_image)
                    SSIM = SSIM_cal(test_my_image, test_sharp_image, val=1.0)
                    # PSNR = PSNR1(test_my_image, test_sharp_image)
                    # SSIM = ssim_l1(test_my_image, test_sharp_image)
                    my_PSNR += PSNR
                    MY_SSIM += SSIM
                my_SSIM_mean = MY_SSIM / len(my_img_shape)
                my_PSNR_mean = my_PSNR / len(my_img_shape)
                print(f"My_{n}:  PSNR:{my_PSNR_mean} ; SSIM:{my_SSIM_mean}")
            ALL_PSNR.append(my_PSNR_mean)
            ALL_SSIM.append(my_SSIM_mean)
        ALL_PSNR = np.mean(ALL_PSNR)
        ALL_SSIM = np.mean(ALL_SSIM)
        print(f"My_Net:PSNR_mean:{ALL_PSNR}; SSIM_mean:{ALL_SSIM}")
        '''
        img = "G:\\ESTRNN\\ESTRNN-master\\result\\GOPR0854_11_00\\00000007_estrnn.png"
        test_deblur_image = cv2.imread(img)
        test_deblur_image = load_images(test_deblur_image)
        ig_blur = "G:\\ESTRNN\\ESTRNN-master\\result\\GOPR0854_11_00\\00000007_input.png"
        test_blur_image = cv2.imread(ig_blur)
        test_blur_image = load_images(test_blur_image)
        '''
        '''
        sharp_image = "G:\\undergraduate\\dataset\\BSD_2ms16ms\\test\\129\\Sharp\\RGB\\00000003.png"
        sharp_img = cv2.imread(sharp_image)
        sharp_img = load_images(sharp_img)
        psnr = PSNR1(test_he_image, sharp_img)
        ssim = ssim_l1(test_he_image, sharp_img)
        psnr_blur = PSNR1(test_blur_image, sharp_img)
        ssim_blur = ssim_l1(test_blur_image, sharp_img)
        PSNR = PSNR1(test_my_image, sharp_img)
        SSIM = ssim_l1(test_my_image, sharp_img)
        print(f"blur: PSNR:{psnr_blur}; SSIM:{ssim_blur}")
        print(f"he_net: PSNR:{psnr}; SSIM:{ssim}")
        print(f'My_Net:PSNR:{PSNR}; SSIM:{SSIM}')
        '''
        '''
        # scrnn
        scrnn_image = "G:\\scrnn\\SRCNN-PyTorch-main\\SRCNN-PyTorch-main\\figure\\129\\00000002_input.png"
        scrnn_image = cv2.imread(scrnn_image)
        scrnn_image = load_images(scrnn_image)
        psnr_scrnn = PSNR1(scrnn_image, sharp_img)
        ssim_scrnn = ssim_l1(scrnn_image, sharp_img)
        print(f"SCRNN_net: PSNR:{psnr_scrnn}; SSIM:{ssim_scrnn}")
        '''


def SSIM_cal(x, y, val=1.0):
    ssim = SSIM_1(y, x, multiprocessing=True, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=val)
    return ssim
class PSNR_cal(_Loss):
    def __init__(self, centralize=True, normalize=True, val_range=255.):
        super(PSNR_cal, self).__init__()
        self.centralize = centralize
        self.normalize = normalize
        self.val_range = val_range

    def _quantize(self, img):
        img = normalize_reverse(img, centralize=self.centralize, normalize=self.normalize, val_range=self.val_range)
        img = img.clamp(0, self.val_range).round()
        return img

    def forward(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        diff = self._quantize(x) - self._quantize(y)
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.val_range).pow(2).view(n, -1).mean(dim=-1)
        psnr = -10 * mse.log10()

        return psnr.mean()
def metrics1(my_img1):
    name = ["015", "020", "028", "044", "052", "053", "069", "072", "073", "075", "078", "080", "099", "107", "109",
            "119", "120", "121", "128", "129"]
    # he_img = "G:\\ESTRNN\\ESTRNN-master\\result\\2ms16ms\\015\\DeBlur"
    # my_image_file_path = "./my_image"  # 增强后图像保存的地址
    # he_img, img_shape = read_load_image2(he_img)
    '''
    test_he_image = he_img[2]  # 这里我选择了第1张照片
    test_blur_image = he_img[3]
    '''
    psnr_cal = PSNR_cal()
    for n in name:
        # my_img = f"save_video/other_epoch_164/{n}/deblur"  {n}/deblur
        my_img = join(my_img1, f"{n}/deblur")
        my_img, my_img_shape = read_load_image2(my_img)
        # blur_img = "save_video/other_epoch_112/015/Blur"
        # blur_img, blur_img_shape = read_load_image2(blur_img)
        sharp_image = f"epoch_329/{n}/GT"
        sharp_image, _ = read_load_image2(sharp_image)
        '''
        test_my_image = my_img[1]
        '''
        if len(my_img_shape) == len(my_img_shape):
            my_PSNR = 0.0
            MY_SSIM = 0.0
            for i in range(len(my_img_shape)):
                test_my_image = my_img[i]
                test_sharp_image = sharp_image[i]
                PSNR = psnr_cal(test_my_image, test_sharp_image)
                # SSIM = ssim_l1(test_my_image, test_sharp_image)
                SSIM = SSIM_cal(test_my_image, test_sharp_image, val=1.0)
                my_PSNR += PSNR
                MY_SSIM += SSIM
            my_SSIM_mean = MY_SSIM / len(my_img_shape)
            my_PSNR_mean = my_PSNR / len(my_img_shape)
            print(f"My_{n}:  PSNR:{my_PSNR_mean} ; SSIM:{my_SSIM_mean}")
    '''
    img = "G:\\ESTRNN\\ESTRNN-master\\result\\GOPR0854_11_00\\00000007_estrnn.png"
    test_deblur_image = cv2.imread(img)
    test_deblur_image = load_images(test_deblur_image)
    ig_blur = "G:\\ESTRNN\\ESTRNN-master\\result\\GOPR0854_11_00\\00000007_input.png"
    test_blur_image = cv2.imread(ig_blur)
    test_blur_image = load_images(test_blur_image)
    '''
    '''
    sharp_image = "G:\\undergraduate\\dataset\\BSD_2ms16ms\\test\\129\\Sharp\\RGB\\00000003.png"
    sharp_img = cv2.imread(sharp_image)
    sharp_img = load_images(sharp_img)
    psnr = PSNR1(test_he_image, sharp_img)
    ssim = ssim_l1(test_he_image, sharp_img)
    psnr_blur = PSNR1(test_blur_image, sharp_img)
    ssim_blur = ssim_l1(test_blur_image, sharp_img)
    PSNR = PSNR1(test_my_image, sharp_img)
    SSIM = ssim_l1(test_my_image, sharp_img)
    print(f"blur: PSNR:{psnr_blur}; SSIM:{ssim_blur}")
    print(f"he_net: PSNR:{psnr}; SSIM:{ssim}")
    print(f'My_Net:PSNR:{PSNR}; SSIM:{SSIM}')
    '''
    '''
    # scrnn
    scrnn_image = "G:\\scrnn\\SRCNN-PyTorch-main\\SRCNN-PyTorch-main\\figure\\129\\00000002_input.png"
    scrnn_image = cv2.imread(scrnn_image)
    scrnn_image = load_images(scrnn_image)
    psnr_scrnn = PSNR1(scrnn_image, sharp_img)
    ssim_scrnn = ssim_l1(scrnn_image, sharp_img)
    print(f"SCRNN_net: PSNR:{psnr_scrnn}; SSIM:{ssim_scrnn}")
    '''


def file_path_gopro(C):
    logger = RecordLogger(".\\record", 'test', 'GOPRO')
    logger.writer = SummaryWriter(logger.save_path)
    # 加载模型参数
    CKPT_PATH = C
    # 加载分解网络模型
    checkpoints = torch.load(CKPT_PATH)
    decom_checkpoint = checkpoints['state_dict']
    e = checkpoints['epoch']
    new_decom = OrderedDict()
    for k, v in decom_checkpoint.items():
        name_1 = k[7:]  # remove `module.`
        new_decom[name_1] = v
    model = ESTRNN().cuda()
    model.load_state_dict(new_decom)
    model.eval()
    dataset_path = "F:\\dataset\\test"
    save_dir_path = f'.\\save_GOPRO_video\\epoch_{e}'
    seqs = sorted(os.listdir(dataset_path))
    for seq in seqs:
        save_dir = join(save_dir_path, seq, "deblur")
        os.makedirs(save_dir, exist_ok=True)
        deblur_save_img_path = join(save_dir, "deblur")
        os.makedirs(deblur_save_img_path, exist_ok=True)
        suffix = 'png'
        start = 0
        test_frames = 10
        end = test_frames
        blur_temp_path = join(dataset_path, seq, "blur")
        img_name = sorted(os.listdir(blur_temp_path))
        seq_length = len(img_name)
        while True:
            input_seq = []
            for frame_idx in range(start, end):
                blur_img_path = join(blur_temp_path, '{}'.format(img_name[frame_idx]))
                blur_img = cv2.imread(blur_img_path).transpose(2, 0, 1)[np.newaxis, ...]
                input_seq.append(blur_img)
            input_seq = torch.tensor(input_seq).permute([1, 0, 2, 3, 4])
            model.eval()
            with torch.no_grad():
                input_seq = torch.tensor(load_images(input_seq.float())).cuda()
                output_seq = model(input_seq)
                if isinstance(output_seq, (list, tuple)):
                    output_seq = output_seq[0]
                output_seq = output_seq.squeeze(dim=0)
            for frame_idx in range(2, end - start - 2):
                deblur_img = output_seq[frame_idx - 2].cpu()
                deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                deblur_img = np.clip(deblur_img * 255.0, 0, 255.0).astype('uint8')
                deblur_img_path_one = join(deblur_save_img_path, '{:08d}_{}.{}'.format(frame_idx + start, 'net_model', suffix))
                cv2.imwrite(deblur_img_path_one, deblur_img)

            if end == seq_length:
                break
            else:
                start = end - 2 - 2
                end = start + test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - test_frames


def metrics_gopro(my_img1):
        seqs = sorted(os.listdir(my_img1))
        ALL_PSNR, ALL_SSIM = [], []
        psnr_cal = PSNR_cal()
        for n in seqs:
            my_img_path = join(my_img1, f"{n}")
            my_img, my_img_shape = read_load_image2(my_img_path)
            # sharp_image_path = f"F:/dataset/test/{n}/Sharp"
            sharp_image_path = f"F:\\dataset\\test\\{n}\\sharp"
            sharp_image, _ = read_load_image2(sharp_image_path)
            my_PSNR = 0.0
            MY_SSIM = 0.0
            for i in range(len(my_img_shape)):
                test_my_image = my_img[i]
                test_sharp_image = sharp_image[i+2]
                # PSNR = PSNR1(test_my_image, test_sharp_image)
                # SSIM = ssim_l1(test_my_image, test_sharp_image)
                PSNR = psnr_cal(test_my_image, test_sharp_image)
                SSIM = SSIM_cal(test_my_image, test_sharp_image, val=1.0)
                my_PSNR += PSNR
                MY_SSIM += SSIM
            my_SSIM_mean = MY_SSIM / len(my_img_shape)
            my_PSNR_mean = my_PSNR / len(my_img_shape)
            print(f"GOPRO_Blur_{n}:  PSNR:{my_PSNR_mean} ; SSIM:{my_SSIM_mean}")
            ALL_PSNR.append(my_PSNR_mean)
            ALL_SSIM.append(my_SSIM_mean)
        print("GOPRO:")
        print(f"Blur:PSNR:{np.mean(ALL_PSNR)}; SSIM: {np.mean(ALL_SSIM)}")


def metrics_signal(path_img):
    my_img1 = join(path_img, "my_net")
    he_img1 = join(path_img, "estrnn")
    sharp_img1 = join(path_img, "sharp")
    blur_img1 = join(path_img, "blur")
    he_img, img_shape = read_load_image2(he_img1)
    my_img, my_img_shape = read_load_image2(my_img1)
    blur_img, _ = read_load_image2(blur_img1)
    sharp_img, _ = read_load_image2(sharp_img1)
    for i in range(len(my_img_shape)):
        test_my_image = my_img[i]
        test_he_image = he_img[i]
        test_blur_image = blur_img[i]
        test_sharp_image = sharp_img[i]
        PSNR_my = PSNR1(test_my_image, test_sharp_image)
        SSIM_my = ssim_l1(test_my_image, test_sharp_image)
        PSNR_he = PSNR1(test_he_image, test_sharp_image)
        SSIM_he = ssim_l1(test_he_image, test_sharp_image)
        PSNR_blur = PSNR1(test_blur_image, test_sharp_image)
        SSIM_blur = ssim_l1(test_blur_image, test_sharp_image)
        print(f"My_{i}:  PSNR:{PSNR_my} ; SSIM:{SSIM_my}")
        print(f"He_{i}:  PSNR:{PSNR_he} ; SSIM:{SSIM_he}")
        print(f"blur_{i}:  PSNR:{PSNR_blur} ; SSIM:{SSIM_blur}")
        print(f"best_{i}: Blur: PSNR:{PSNR_my-PSNR_blur}; SSIM:{SSIM_my-SSIM_blur};\n He:PSNR:{PSNR_my-PSNR_he}; SSIM:{SSIM_my-SSIM_he}")


# metrics_signal("I/854-11-00")
# file_path_BSD("MyNet_569_best.pkl")
metrics("save_video/epoch_569")
file_path_gopro("MyNet_569_best.pkl")
metrics_gopro("save_GOPRO_video/epoch_569")
name = ["163", "167", "173"]
for n in name:
    CKPT_PATH = f"MyNet_{n}_best.pkl"
    file_path_main(CKPT_PATH)
    my_img = f"save_video/epoch_{n}"
    print(f"{n}:\n")
    metrics(my_img)
