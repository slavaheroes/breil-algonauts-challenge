# https://github.com/openai/CLIP
import os
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
# import clip
import timm


data_dir = '/SSD/qasymjomart/algonauts/data'
save_dir = '/SSD/qasymjomart/algonauts/breil-algonauts-challenge/eva/eva_features'

if __name__ == "__main__":
    device = 'cuda:0'
    # model, preprocess = clip.load("ViT-L/14", device=device)
    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True).to(device)
    model.head = nn.Identity()
    transform = transforms.Compose([
        transforms.Resize((448,448)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])
    
    for subj in range(1, 9):
        subj_dir = os.path.join(data_dir, 'subj'+format(subj, '02'))
        train_imgs_paths = os.path.join(subj_dir, 'training_split', 'training_images')
        train_imgs = os.listdir(train_imgs_paths)
        
        test_imgs_paths = os.path.join(subj_dir, 'test_split', 'test_images')
        test_imgs = os.listdir(test_imgs_paths)
        
        # features savenames
        save_dir_train = os.path.join(save_dir, 'subj'+format(subj, '02'), 'train_features')
        os.makedirs(save_dir_train, exist_ok=True)
        
        save_dir_test = os.path.join(save_dir, 'subj'+format(subj, '02'), 'test_features')
        os.makedirs(save_dir_test, exist_ok=True)

        for train_img in tqdm(train_imgs, desc=f'subj {subj} train'):
            
            image = transform(
                Image.open(os.path.join(train_imgs_paths, train_img)).convert('RGB')
            ).unsqueeze(0)
            
            with torch.no_grad():
                image_features = model(image.to(device))
                
            image_features = image_features.cpu().squeeze().detach().numpy()

            # save features
            with open(
                os.path.join(save_dir_train, train_img[:-4] + ".npy"), 'wb'
            ) as f:
                np.save(f, image_features)
        
        for test_img in tqdm(test_imgs, desc=f'subj {subj} test'):
            
            image = transform(
                Image.open(os.path.join(test_imgs_paths, test_img)).convert('RGB')
            ).unsqueeze(0)
            
            with torch.no_grad():
                image_features = model(image.to(device))
                
            image_features = image_features.cpu().squeeze().detach().numpy()

            # save features
            with open(
                os.path.join(save_dir_test, test_img[:-4] + ".npy"), 'wb'
            ) as f:
                np.save(f, image_features)