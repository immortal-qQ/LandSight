import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image
import cv2

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class PersonalPhotoDataset(Dataset):

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = A.Compose([
            ToTensorV2(p=1.0)
        ])
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                     if image[-4:] == '.jpg']
        # print(self.imgs)
        # classes: 0 index is reserved for background
        self.classes = ['_', 'EiffelTower', 'PisaTower', 'StatueOfLiberty']

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        # print(f'Selected: {img_name}')
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        sample = self.transforms(image=img_res)

        img_res = sample['image']

        return img_res

    def __len__(self):
        return len(self.imgs)


def plot_img_bbox(img, target, ax, classes):
    ax.imshow(img)
    for i, box in enumerate(target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none',
                                 # label= test_dataset.classes[target['labels'][i].item()]
                                 )

        ax.add_patch(rect)
        ax.text(x, y, classes[target['labels'][i].item()], color='r', fontsize='large')


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    return final_prediction


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')
