import math
import string

import torch
import torch.nn as nn
import numpy as np
import cv2
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import random

from PIL import Image
import matplotlib.pyplot as plt


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    Img = np.transpose(Img, (0, 2, 3, 1))
    Iclean = np.transpose(Iclean, (0, 2, 3, 1))
    # print(Iclean.shape)
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range,
                             channel_axis=-1)
    return (SSIM / Img.shape[0])


def batch_RMSE(img, imclean, data_range):
    img = img * 255
    imclean = imclean * 255
    Img = img.data.cpu().numpy().astype(np.uint8)

    Iclean = imclean.data.cpu().numpy().astype(np.uint8)
    MSE = 0
    for i in range(Img.shape[0]):
        MSE += math.sqrt(compare_mse(Iclean[i, :, :, :], Img[i, :, :, :]))
    return (MSE / Img.shape[0])
def multi( img_train ):
    random_img = random.randint(1, 12)
"""

def add_watermark_noise(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    random_img = random.randint(1, 12)
    # 对比实验的时候选取某个水印进行去除
    # random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path =  "/mnt/sda/zhouying/NewData/promptsmall"
    #data_path = "watermark/translucence/"
    watermark = Image.open(data_path + '/'+str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    alpha = random.uniform(0,1)
    alpha=1
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                # color = color[::-1]
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)
    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))
    print(random_img)
    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        sum=0
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.3, 0.6)
            # 原本的是（0.5，1）
            # scale = 1.5
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            #下面是原版，但是
            #x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            #y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            x = random.randint(0, img_w - int(w * scale)+ 400)  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale)+ 400)  # int(-h * scale)
            #print(x,y)
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            #img_cnt = np.array(img_for_cnt)
            #sum = (img_cnt > 0).sum()

            sum1 =random.randint(100,800)
            sum = sum +sum1
            ratio = img_w * img_h * occupancy / 600
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float64) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train

"""
def add_watermark_noise(img_train, occupancy=50, self_surpervision=False, same_random=1, alpha=0.3):

    # 转换输入为NumPy数组并获取维度

    img_train_np = img_train.numpy()

    batch_size, channels, img_h, img_w = img_train_np.shape  # [N, C, H, W]

    

    # 初始化掩码 (修正维度为 [Batch, 1, H, W])

    mask_all = np.zeros((batch_size, 1, img_h, img_w), dtype=np.uint8)


    # 加载水印并设置透明度

    random_img = random.randint(1, 2)
    # 对比实验的时候选取某个水印进行去除
    # random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random

    data_path = "/mnt/sda/zhouying/NewData/strip"

    watermark = Image.open(f"{data_path}/{random_img}.png").convert("RGBA")

    w, h = watermark.size


    # 调整水印透明度

    watermark_alpha = Image.new("RGBA", watermark.size)

    for x in range(w):

        for y in range(h):

            r, g, b, a = watermark.getpixel((x, y))

            new_alpha = int(255 * alpha) if a != 0 else 0

            watermark_alpha.putpixel((x, y), (r, g, b, new_alpha))

    watermark = watermark_alpha


    # 转换图像格式为 NHWC (便于PIL处理)

    img_train_nhwc = np.transpose(img_train_np, (0, 2, 3, 1))  # [N, H, W, C]


    for i in range(batch_size):

        img_pil = Image.fromarray((img_train_nhwc[i] * 255).astype(np.uint8)).convert("RGBA")

        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        occupancy_target = np.random.uniform(0, occupancy)


        while True:

            # 随机变换参数

            angle = random.randint(-45, 45)

            scale = np.random.uniform(0.3, 0.6)

            scaled_watermark = watermark.resize((int(w * scale), int(h * scale))).rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

            water_width, water_height = scaled_watermark.size


            # 确保水印位置在图像范围内 (关键修正)

            x = random.randint(0, max(1, img_w - water_width))

            y = random.randint(0, max(1, img_h - water_height))


            # 提取Alpha通道生成掩码

            water_alpha = np.array(scaled_watermark.split()[3])

            water_mask = (water_alpha > 0).astype(np.uint8)


            # 计算有效区域

            paste_x_end = min(x + water_width, img_w)

            paste_y_end = min(y + water_height, img_h)

            crop_x_start = max(0, -x)

            crop_x_end = water_width - max(0, (x + water_width) - img_w)

            crop_y_start = max(0, -y)

            crop_y_end = water_height - max(0, (y + water_height) - img_h)


            # 裁剪掩码

            water_crop = water_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            target_region = mask[y:paste_y_end, x:paste_x_end]


            # 强制形状一致

            min_h = min(target_region.shape[0], water_crop.shape[0])

            min_w = min(target_region.shape[1], water_crop.shape[1])

            water_crop = water_crop[:min_h, :min_w]

            target_region = target_region[:min_h, :min_w]


            # 更新掩码

            mask[y:paste_y_end, x:paste_x_end] = np.maximum(target_region, water_crop)


            # 粘贴水印

            layer = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))

            layer.paste(scaled_watermark, (x, y))

            img_pil = Image.composite(layer, img_pil, layer)

            
            # 检查覆盖率

            coverage = np.sum(mask) / (img_h * img_w) * 100

            if coverage >= occupancy_target:

                break


        # 转换为RGB并归一化 (关键修正)

        img_rgb = np.array(img_pil.convert("RGB"))  # 确保输出为3通道

        img_train_nhwc[i] = img_rgb.astype(np.float32) / 255.0

        mask_all[i, 0] = mask  # 掩码维度 [1, H, W]


    # 转换回原始格式 [N, C, H, W]

    img_train_np = np.transpose(img_train_nhwc, (0, 3, 1, 2))

    return torch.from_numpy(img_train_np), torch.from_numpy(mask_all)

def add_watermark_noise_B(img_train, occupancy=20, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "watermark/translucence/"
    watermark = Image.open(data_path + str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    alpha = 0.3 + random.randint(0, 70) * 0.01
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                # color = color[::-1]
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.5, 1.0)
            # scale = 1.5
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train




#  这个函数只用来测试
def add_watermark_noise_test(img_train, occupancy=50, img_id=1, scale_img=1.5, self_surpervision=False,
                                same_random=0, alpha=30.):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    print(img_id)
    random_img = img_id  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "/mnt/sda/zhouying/NewData/strip/"
    watermark = Image.open(data_path + str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)  # random.randint(100)
                # color = color[::-1]
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.3, 1.0)
            scale = scale_img
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            #print(img_w,w)
            #下面这两句 原作者也没写好
            # x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            #y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            x = 128
            y = 128
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float64) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train


import torchvision.models as models
from models import VGG16


def load_froze_vgg16():
    # finetunning
    model_pretrain_vgg = models.vgg16(pretrained=True)

    # load VGG16
    net_vgg = VGG16()
    model_dict = net_vgg.state_dict()
    pretrained_dict = model_pretrain_vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # load parameters
    net_vgg.load_state_dict(pretrained_dict)

    for child in net_vgg.children():
        for p in child.parameters():
            p.requires_grad = False
    device_ids = [0]

    #model_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).cuda()
    model_vgg = net_vgg.cuda()


    return model_vgg

def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
import yaml


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
