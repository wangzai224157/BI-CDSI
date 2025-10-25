#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-7 20:23
# @Author  : 26731
# @File    : test_gswm.py
# @Software: PyCharm
import cv2
import os
import argparse
import glob
from torch.autograd import Variable
from hint.networks import HINT,HINT1
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as matImage




# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
config = get_config('configs/config.yaml')

parser = argparse.ArgumentParser(description="watermark removal")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='path of model files')  # /media/npu/Data/jtc/data/models
parser.add_argument("--net", type=str, default="HN", help='Network used in test')
parser.add_argument("--test_data", type=str, default='/mnt/sda/zhouying/NewData/SAFE/SAFE', help='The set of tests we created')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--display", type=str, default="False", help='Whether to display an image')
parser.add_argument("--test_path", type=str, default="/mnt/sda/zhouying/NewData/SAFE/SAFE", help='The loss function used for training')
parser.add_argument("--alpha", type=float, default=1.0, help="The opacity of the watermark")
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=str, default="True", help='Whether to use perception network')
parser.add_argument("--output_path", type=str, default="/mnt/sda/zhouying/2.48/SWCNN-main_multi/result", help='result')
parser.add_argument("--pth1", type=str, default="/mnt/sda/zhouying/2.48/SWCNN-main_multi/pixel_pth/HNperL1n2nalpha1.0.pth", help='model1.path')

opt = parser.parse_args()

if opt.PN == "True":
    model_name_1 = "per"
else:
    model_name_1 = "woper"
if opt.loss == "L1":
    model_name_2 = "L1"
else:
    model_name_2 = "L2"
if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"
tensorboard_name = opt.net + model_name_1 + model_name_2 + model_name_3 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"


def normalize(data):
    return data / 255.

"""
def water_test():
    # Build model
    print('Loading model ...\n')
    if opt.net == "HN":
        net1 = HINT1()
        net = HINT()
    else:
        assert False
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # load model
    #model = net.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.modeldir, model_name)))
    model1 = nn.DataParallel(net1, device_ids=device_ids).cuda()
    model.eval()
    print('Loading data info ...\n')
    #data_path = config['train_data_path']  # load test data'/media/npu/Data/jtc/data/'
    data_path = os.path.join(opt.test_path, '*.jpg')
    files_source = glob.glob(data_path)
    files_source.sort()
    test_len=len(files_source)
    # process data
    all_psnr_source_avg = 0
    all_ssim_source_avg = 0
    all_mse_source_avg = 0

    all_psnr_avg = 0
    all_ssim_avg = 0
    all_mse_avg = 0
    for img_index in range(12):
        img_index += 1
        psnr_test = 0
        f_index = 0
        ssim_test = 0
        mse_test = 0
        psnr_source_avg = 0
        ssim_source_avg = 0
        mse_source_avg = 0
        
        for f in files_source:
            # image
            Img = cv2.imread(f)
            try:
                Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                matImage.imsave(opt.output_path + "/clear_out" + str(f_index) + ".jpg", Img_rgb)
            except Exception as e:
                print(e)
            Img = normalize(np.float32(Img[:, :, :]))
            Img = np.expand_dims(Img, 0)
            # Img = np.expand_dims(Img, 1)
            Img = np.transpose(Img, (0, 3, 1, 2))
            _, _, w, h = Img.shape
            w = int(int(w / 32) * 32)
            h = int(int(h / 32) * 32)
            Img = Img[:, :, 0:w, 0:h]
            ISource = torch.Tensor(Img)
            # noise
            noise_gs = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
            # add watermark
            INoisy  = add_watermark_noise_test(ISource, 0., img_id=1, scale_img=1.5, alpha=opt.alpha)
            INoisy = torch.Tensor(INoisy)  # + noise_gs
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            with torch.no_grad():  # this can save much memory
                if opt.net == "FFDNet":
                    noise_sigma = 0 / 255.
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(INoisy.shape[0])]))
                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.cuda()
                    Out = torch.clamp(model(INoisy, noise_sigma), 0., 1.)
                else:

                    ckpt = torch.load(opt.pth1)
                    model.load_state_dict(ckpt,strict = False) 
                    imgn_test1 = torch.cat((INoisy, INoisy, INoisy, INoisy), dim=1)
                    out  = model(imgn_test1)
                    Out = torch.clamp(out, 0., 1.)
                    print(Out.size())
                    #Out  = Out.permute(0, 2, 1)
                    #out_mask = torch.clamp(out_mask, 0., 1.)
                INoisy = torch.clamp(INoisy, 0., 1.)

            if opt.display == "True":
                Out  = torch.cat((Out, Out, Out),dim = 1)
                Out_np = Out.cpu().numpy()
                INoisy_np = INoisy.cpu().numpy()
                #Out_np = torch.cat((Out_np,Out_np,Out_np),dim = 1)
                #out_mask  = out_mask.cpu().numpy() 
                # print(Out_np)
                pic = Out_np[0]
                
                r, g, b = pic[0], pic[1], pic[2]
                #r, g, b = pic, pic, pic
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                
                pic = np.transpose(pic, (1, 2, 0))
                
                plt.subplot(121)
                plt.imshow(pic)
                matImage.imsave(opt.output_path + "/pic_out" + str(f_index) + ".jpg", pic)
                #plt.subplot(122)
                pic = INoisy_np[0]
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                #plt.imshow(pic)
                matImage.imsave(opt.output_path + "/pic_input" + str(f_index) + ".jpg", pic)

                
                f_index += 1
                #plt.show()


            psnr_source = batch_PSNR(INoisy, ISource, 1.0)
            ssim_source = batch_SSIM(INoisy, ISource, 1.0)
            mse_source = batch_RMSE(INoisy, ISource, 1.0)
            psnr_api = batch_PSNR(Out, ISource, 1.)
            ssim_api = batch_SSIM(Out, ISource, 1.)
            mse_api = batch_RMSE(Out, ISource, 1.)
            psnr_test += psnr_api
            ssim_test += ssim_api
            mse_test += mse_api
            psnr_source_avg += psnr_source
            ssim_source_avg += ssim_source
            mse_source_avg += mse_source
            print("%s PSNR_API %f SSIM_API %f MSE_API %f" % (f, psnr_api, ssim_api, mse_api))
            print("%s PSNR_Noise %f SSIM_Noise %f MSE_Noise %f" % (f, psnr_source, ssim_source, mse_source))
        psnr_test /= len(files_source)
        ssim_test /= len(files_source)
        mse_test /= len(files_source)
        psnr_source_avg /= len(files_source)
        ssim_source_avg /= len(files_source)
        mse_source_avg /= len(files_source)
        print("\nPSNR on test data %f SSIM on test data %f MSE on test data %f" % (psnr_test, ssim_test, mse_test))
        print("\nPSNR on Noisy data %f SSIM on Noisy data %f MSE on Noisy data %f" % (
            psnr_source_avg, ssim_source_avg, mse_source_avg))

        all_psnr_avg += psnr_test
        all_mse_avg += mse_test
        all_ssim_avg += ssim_test

        all_psnr_source_avg += psnr_source_avg
        all_mse_source_avg += mse_source_avg
        all_ssim_source_avg += ssim_source_avg

    all_ssim_source_avg /= 12
    all_mse_source_avg /= 12
    all_psnr_source_avg /= 12

    all_ssim_avg /= 12
    all_mse_avg /= 12
    all_psnr_avg /= 12
    print("\nALL_PSNR on test data %f SSIM on test data %f MSE on test data %f" % (
    all_psnr_avg, all_ssim_avg, all_mse_avg))
    print("\nALL_PSNR on Noisy data %f ALL_SSIM on Noisy data %f ALL_MSE on Noisy data %f" % (
        all_psnr_source_avg, all_ssim_source_avg, all_mse_source_avg))

"""



def water_test():
    # Build model
    print('Loading model ...\n')
    if opt.net == "HN":
        net1 = HINT1()
        net = HINT()
    else:
        assert False
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # load model
    #model = net.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.modeldir, model_name)))
    model1 = nn.DataParallel(net1, device_ids=device_ids).cuda()
    model.eval()
    print('Loading data info ...\n')
    #data_path = config['train_data_path']  # load test data'/media/npu/Data/jtc/data/'
    data_path = os.path.join(opt.test_path, '*.jpg')
    files_source = glob.glob(data_path)
    files_source.sort()
    test_len=len(files_source)
    # process data
    all_psnr_source_avg = 0
    all_ssim_source_avg = 0
    all_mse_source_avg = 0

    all_psnr_avg = 0
    all_ssim_avg = 0
    all_mse_avg = 0
    for img_index in range(12):
        img_index += 1
        psnr_test = 0
        f_index = 0
        ssim_test = 0
        mse_test = 0
        psnr_source_avg = 0
        ssim_source_avg = 0
        mse_source_avg = 0
        
        for f in files_source:
            
            # 读取图像：处理4通道RGBA图像，剥离Alpha通道
            Img = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # 用IMREAD_UNCHANGED读取所有通道（包括Alpha）
             # 关键修复1：处理4通道RGBA，转为3通道RGB
            print("Img.shape")
            print(Img.shape)
            if Img.shape[-1] == 4:  # 若为4通道（RGBA）
                Img = Img[:, :, :3]  # 只保留前3个RGB通道，丢弃Alpha通道
            elif Img.shape[-1] != 3:  # 若不是3通道（如1通道灰度图）
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)  # 转为3通道BGR，统一格式
        
            try:
                Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # 3通道BGR转RGB
                matImage.imsave(opt.output_path + "/clear_out" + str(f_index) + ".jpg", Img_rgb)
            except Exception as e:
                print(e)
                try:
                    Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                    matImage.imsave(opt.output_path + "/clear_out" + str(f_index) + ".jpg", Img_rgb)
                except Exception as e:
                    print(e)
            # 后续预处理保持不变，但此时Img已确保是3通道
            Img = normalize(np.float32(Img[:, :, :]))
            Img = np.expand_dims(Img, 0)
            Img = np.transpose(Img, (0, 3, 1, 2))  # 此时通道数为3，张量形状(1, 3, h, w)
            _, _, w, h = Img.shape
            w = int(int(w / 32) * 32)
            h = int(int(h / 32) * 32)
            Img = Img[:, :, 0:w, 0:h]
            ISource = torch.Tensor(Img)



            
            # noise
            noise_gs = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
            # add watermark
            INoisy  = add_watermark_noise_test(ISource, 0., img_id=1, scale_img=1.5, alpha=opt.alpha)
            INoisy = torch.Tensor(INoisy)  # + noise_gs
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            with torch.no_grad():  # this can save much memory
                if opt.net == "FFDNet":
                    noise_sigma = 0 / 255.
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(INoisy.shape[0])]))
                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.cuda()
                    Out = torch.clamp(model(INoisy, noise_sigma), 0., 1.)
                else:

                    ckpt = torch.load(opt.pth1)
                    model.load_state_dict(ckpt,strict = False) 
                    imgn_test1 = torch.cat((INoisy, INoisy, INoisy, INoisy), dim=1)
                    out  = model(imgn_test1)
                    Out = torch.clamp(out, 0., 1.)
                    print(Out.size())
                    #Out  = Out.permute(0, 2, 1)
                    #out_mask = torch.clamp(out_mask, 0., 1.)
                INoisy = torch.clamp(INoisy, 0., 1.)

            if opt.display == "True":
                if Out.shape[1] == 1:  # 若模型输出为1通道（灰度）
                    Out = torch.cat((Out, Out, Out), dim=1)  # 转为3通道RGB
                
                Out_np = Out.cpu().numpy()
                INoisy_np = INoisy.cpu().numpy()
                #Out_np = torch.cat((Out_np,Out_np,Out_np),dim = 1)
                #out_mask  = out_mask.cpu().numpy() 
                # print(Out_np)
                pic = Out_np[0]
                
                r, g, b = pic[0], pic[1], pic[2]
                #r, g, b = pic, pic, pic
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                
                pic = np.transpose(pic, (1, 2, 0))
                
                plt.subplot(121)
                plt.imshow(pic)
                matImage.imsave(opt.output_path + "/pic_out" + str(f_index) + ".jpg", pic)
                #plt.subplot(122)
                pic = INoisy_np[0]
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                #plt.imshow(pic)
                matImage.imsave(opt.output_path + "/pic_input" + str(f_index) + ".jpg", pic)

                
                f_index += 1
                #plt.show()


            psnr_source = batch_PSNR(INoisy, ISource, 1.0)
            ssim_source = batch_SSIM(INoisy, ISource, 1.0)
            mse_source = batch_RMSE(INoisy, ISource, 1.0)
            psnr_api = batch_PSNR(Out, ISource, 1.)
            ssim_api = batch_SSIM(Out, ISource, 1.)
            mse_api = batch_RMSE(Out, ISource, 1.)
            psnr_test += psnr_api
            ssim_test += ssim_api
            mse_test += mse_api
            psnr_source_avg += psnr_source
            ssim_source_avg += ssim_source
            mse_source_avg += mse_source
            print("%s PSNR_API %f SSIM_API %f MSE_API %f" % (f, psnr_api, ssim_api, mse_api))
            print("%s PSNR_Noise %f SSIM_Noise %f MSE_Noise %f" % (f, psnr_source, ssim_source, mse_source))
        psnr_test /= len(files_source)
        ssim_test /= len(files_source)
        mse_test /= len(files_source)
        psnr_source_avg /= len(files_source)
        ssim_source_avg /= len(files_source)
        mse_source_avg /= len(files_source)
        print("\nPSNR on test data %f SSIM on test data %f MSE on test data %f" % (psnr_test, ssim_test, mse_test))
        print("\nPSNR on Noisy data %f SSIM on Noisy data %f MSE on Noisy data %f" % (
            psnr_source_avg, ssim_source_avg, mse_source_avg))

        all_psnr_avg += psnr_test
        all_mse_avg += mse_test
        all_ssim_avg += ssim_test

        all_psnr_source_avg += psnr_source_avg
        all_mse_source_avg += mse_source_avg
        all_ssim_source_avg += ssim_source_avg

    all_ssim_source_avg /= 12
    all_mse_source_avg /= 12
    all_psnr_source_avg /= 12

    all_ssim_avg /= 12
    all_mse_avg /= 12
    all_psnr_avg /= 12
    print("\nALL_PSNR on test data %f SSIM on test data %f MSE on test data %f" % (
    all_psnr_avg, all_ssim_avg, all_mse_avg))
    print("\nALL_PSNR on Noisy data %f ALL_SSIM on Noisy data %f ALL_MSE on Noisy data %f" % (
        all_psnr_source_avg, all_ssim_source_avg, all_mse_source_avg))

if __name__ == "__main__":
    # main()
    water_test()
