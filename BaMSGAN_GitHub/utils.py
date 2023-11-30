#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import cv2
from skimage import io
import torch
import random

from pathlib import Path
from torchvision.utils import save_image


def img_repo(img, epoch, num):
    Path('data_his').mkdir(parents=True, exist_ok=True)
    save_image(img, os.path.join('data_his', f'epoch{epoch}_num{num}.png'))


def len_repo():
    path = './data_his'
    files = os.listdir(path)
    l = len(files)
    return l


def del_img(k):
    path = './data_his'
    files = os.listdir(path)
    os.remove(os.path.join(path, files[k]))


def rand_del(i):
    l = len_repo()
    if l > 400:
        n = random.sample(range(l), i)
        n.sort(reverse=True)
        for j in range(i):
            del_img(n[j])


def samimg(n):
    image = []
    path = './data_his'
    files = os.listdir(path)
    rand_del(n)
    if os.path.exists('./data_his/.ipynb_checkpoints'):
        os.removedirs(os.path.join(path,'.ipynb_checkpoints'))
    l = len_repo()
    p = random.sample(range(l), n)
    path1 = './data_his'
    for j in range(n):
        files = os.listdir(path)
        b = os.path.join(path1, files[p[j]])
        b = torch.Tensor(cv2.imread(b))
        b = b.permute(2, 0, 1)
        b = b.unsqueeze(0)
        image.append(b)

    return torch.cat(image, dim=0)
