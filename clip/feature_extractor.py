# https://github.com/openai/CLIP
import os
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import clip



data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
save_dir = '/SSD/slava/algonauts/clip_features'

if __name__ == "__main__":
    device = 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    
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
            
            image = preprocess(
                Image.open(os.path.join(train_imgs_paths, train_img))
            ).unsqueeze(0)
            
            with torch.no_grad():
                image_features = model.encode_image(image)
                
            image_features = image_features.cpu().squeeze().detach().numpy()

            # save features
            with open(
                os.path.join(save_dir_train, train_img[:-4] + ".npy"), 'wb'
            ) as f:
                np.save(f, image_features)

        
        for test_img in tqdm(test_imgs, desc=f'subj {subj} test'):
            
            image = preprocess(
                Image.open(os.path.join(test_imgs_paths, test_img))
            ).unsqueeze(0)
            
            with torch.no_grad():
                image_features = model.encode_image(image)
                
            image_features = image_features.cpu().squeeze().detach().numpy()

            # save features
            with open(
                os.path.join(save_dir_test, test_img[:-4] + ".npy"), 'wb'
            ) as f:
                np.save(f, image_features)
        
        
            
            

