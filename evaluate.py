import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time
import math

from network import DIFNet
from args import args

is_cuda = torch.cuda.is_available()

class MyTestDataset(Dataset):
    def __init__(self, img1_path_file, img2_path_file):
        f1 = open(img1_path_file, 'r')
        img1_list = f1.read().splitlines()
        f1.close()
        f2 = open(img2_path_file, 'r')
        img2_list = f2.read().splitlines()
        f2.close()

        self.img1_list = img1_list
        self.img2_list = img2_list
    
    def __getitem__(self, index):
        img1 = Image.open(self.img1_list[index]).convert('RGB')
        img2 = Image.open(self.img2_list[index]).convert('RGB')     
        
        custom_transform_rgb = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                   transforms.Resize((args.HEIGHT,args.WIDTH)),
                                                   transforms.ToTensor()])
        custom_transform_gray = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                    transforms.Resize((args.HEIGHT,args.WIDTH)),
                                                    transforms.ToTensor()])
        
        img1 = custom_transform_rgb(img1)
        img2 = custom_transform_gray(img2)

        return img1, img2

    def __len__(self):

        return len(self.img1_list)


if __name__ == '__main__':

    model = DIFNet()

    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    img1_path_file = args.test_visible
    img2_path_file = args.test_lwir

    testloader = DataLoader(MyTestDataset(img1_path_file,img2_path_file), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if is_cuda:
        model.cuda()

    for i, (img_1, img_2) in enumerate(tqdm(testloader)):

        img_cat = torch.cat([img_1,img_2],dim=1)
        if is_cuda and args.gpu is not None:
            img_1 = img_1.cuda()
            img_2 = img_2.cuda()
            img_cat = img_cat.cuda()

        fusion = model(img_cat)

        if is_cuda:
            fusion = fusion.cpu()
        else:
            pass

        save_image(img_1,args.test_save_dir + '{}_visible.png'.format(i))
        save_image(img_2,args.test_save_dir + '{}_ir.png'.format(i))
        save_image(fusion,args.test_save_dir + '{}_fusion.png'.format(i))
        save_image((img_1 + img_2) / 2,args.test_save_dir + '{}_add.png'.format(i))
        

    print('Finished testing')

    