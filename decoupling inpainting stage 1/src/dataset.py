import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
# from scipy.misc import imread
from skimage.color import rgb2gray
# from scipy.misc import imresize
from .utils import create_mask, create_line_mask, generate_stroke_mask, create_irregular_mask
import cv2
from skimage.feature import canny

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK
        

       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])
    
        if size != 0:
            h_idx = 48
            w_idx = 48
            img = img[h_idx : h_idx + size, w_idx : w_idx + size,:]
            img = self.resize(img, size, size, centerCrop=True)
            



        # load mask
        mask = self.load_mask(img, index)


        return self.to_tensor(img), self.to_tensor(mask)




    def load_lmk(self, target_shape, index, size_before, center_crop = True):
        mask_type = 4
        imgh,imgw = target_shape[0:2]
        landmarks = np.genfromtxt(self.landmark_data[index])
        landmarks = landmarks.reshape(self.config.LANDMARK_POINTS, 2)

        if self.input_size != 0:
            if center_crop:
                side = np.minimum(size_before[0],size_before[1])
                i = (size_before[0] - side) // 2
                j = (size_before[1] - side) // 2
                landmarks[0:self.config.LANDMARK_POINTS , 0] -= j
                landmarks[0:self.config.LANDMARK_POINTS , 1] -= i

            landmarks[0:self.config.LANDMARK_POINTS ,0] *= (imgw/side)
            landmarks[0:self.config.LANDMARK_POINTS ,1] *= (imgh/side)
        landmarks = (landmarks+0.5).astype(np.int16)

        return landmarks



    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        #mask_type = self.mask
        #mask_type = random.randint(0, mask_type )
        mask_type = 2



        # 50% no mask, 25% random block mask, 25% external mask, for landmark predictor training.
        if mask_type == 9:
            mask_type = 0 if np.random.uniform(0,1) >= 0.5 else 4

        # no mask
        #if mask_type == 0:
            #return np.zeros((self.config.INPUT_SIZE,self.config.INPUT_SIZE))


        if mask_type == 0:
            return create_irregular_mask(width=imgw,height=imgh)


        # external + random block
        if mask_type == 8:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # center mask
        if mask_type == 2:
            return create_mask(imgw, imgh, int(imgw * 5/ 16) , int(imgh * 5/ 16), x = imgw//16, y = imgh//16)

        # external
        if mask_type == 7:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            #mask = (mask > 100).astype(np.uint8) * 255
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index%len(self.mask_data)])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            # mask = 1- (mask < 200).astype(np.uint8) * 255
            return mask
        
            # 新增：多线条/倾斜线性缺失区域掩码（类型7）
        if mask_type == 3:
            return create_line_mask(
                width=imgw,
                height=imgh,
                num_lines=random.randint(2, 5),  # 随机2-5条线
                line_width_range=(70, 100),
                angle_range=(0, 180),  # 全角度范围
                gap_range=(15, 40),
                randomize=True
            )
        
        # 新增：固定方向的多线条掩码（类型8）
        if mask_type == 4:
            # 固定为水平或垂直方向
            angle = 0 if random.random() < 0.5 else 90
          
            return create_line_mask(
                width=imgw,
                height=imgh,
                num_lines=random.randint(1, 4),
                line_width_range=(60, 80),
                angle_range=(angle, angle),  # 固定角度
                gap_range=(20, 60),
                randomize=False
            )
        
        """
        if mask_type == 4:
            # 固定为水平或垂直方向
            angle = 0 if random.random() < 0.5 else 90
          
            return create_line_mask(
                width=imgw,
                height=imgh,
                num_lines=random.randint(1,1),
                line_width_range=(60, 60),
                angle_range=(angle, angle),  # 固定角度
                gap_range=(60, 60),
                randomize=False
            )
        """



            # 随机的不规则
        if mask_type == 5:
            # 随机的
            return generate_stroke_mask(width=imgw,height=imgh)
          
           


    """
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    """    
    def to_tensor(self, img):
        img = Image.fromarray(img)  # This should now work since img is uint8
        img = transforms.ToTensor()(img)
        return img

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = scipy.misc.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize((height, width)))
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]
        
        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item




def image_transforms(load_size):

    return transforms.Compose([

        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
