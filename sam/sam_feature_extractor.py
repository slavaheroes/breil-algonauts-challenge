import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from segment_anything import sam_model_registry


transform = transforms.Compose([
    transforms.Resize((1024,1024)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])

batch_size = 1
device = 'cuda:1'
device = torch.device(device)

model = sam_model_registry["vit_l"](checkpoint="/SSD/slava/pretrained_models/sam_vit_l_0b3195.pth").image_encoder
# model = torch.nn.DataParallel(model)
model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
model.eval()


data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
save_dir = '/SSD/slava/algonauts/sam_large_features'
    
if __name__=="__main__":
    
    for subj in range(1, 9):
        subj_dir = os.path.join(data_dir, 'subj'+format(subj, '02'))
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
            ).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model(image)
            
            image_features = image_features.cpu().squeeze().detach().numpy()

            # save features
            with open(
                os.path.join(save_dir_train, train_img[:-4] + ".npy"), 'wb'
            ) as f:
                np.save(f, image_features)
                        
        for test_img in tqdm(test_imgs, desc=f'subj {subj} test'):
            
            image = transform(
                Image.open(os.path.join(test_imgs_paths, test_img)).convert('RGB')
            ).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model(image)
                
            image_features = image_features.cpu().squeeze().detach().numpy()

            # save features
            with open(
                os.path.join(save_dir_test, test_img[:-4] + ".npy"), 'wb'
            ) as f:
                np.save(f, image_features)
                
        
        
        

         