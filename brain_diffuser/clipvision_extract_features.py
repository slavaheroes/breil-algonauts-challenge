import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import torchvision.transforms as T

cfgm_name = 'vd_noema'

pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)

class batch_generator_external_images(Dataset):

    def __init__(self, data_path, filenames):
        self.data_path = data_path
        self.filenames = filenames


    def __getitem__(self,idx):
        path = os.path.join(self.data_path, self.filenames[idx])
        img = Image.open(path)
        img = T.functional.resize(img,(512,512))
        img = T.functional.to_tensor(img).float()
        #img = img/255
        img = img*2 - 1
        return img, self.filenames[idx]

    def __len__(self):
        return  len(self.filenames)

data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
save_dir = '/SSD/slava/algonauts/brain_diffuser_clip_vision_features/'

if __name__=="__main__":
    for subj in range(1, 9):
        subj_dir = os.path.join(data_dir, 'subj'+format(subj, '02'))
        train_imgs_paths = os.path.join(subj_dir, 'training_split', 'training_images')
        train_imgs = sorted(os.listdir(train_imgs_paths))
        train_ds = batch_generator_external_images(
            train_imgs_paths, train_imgs
        )

        test_imgs_paths = os.path.join(subj_dir, 'test_split', 'test_images')
        test_imgs = sorted(os.listdir(test_imgs_paths))
        test_ds = batch_generator_external_images(
            test_imgs_paths, test_imgs
        )
        
        # features savenames
        save_dir_train = os.path.join(save_dir, 'subj'+format(subj, '02'), 'train_features')
        os.makedirs(save_dir_train, exist_ok=True)
        
        save_dir_test = os.path.join(save_dir, 'subj'+format(subj, '02'), 'test_features')
        os.makedirs(save_dir_test, exist_ok=True)

        num_embed, num_features, num_test, num_train = 257, 768, len(test_ds), len(train_ds)

        for img, filename in tqdm(train_ds, desc=f'subj {subj} train'):
            
            with torch.no_grad():
                c = net.clip_encode_vision(img.unsqueeze(0))[0]
                c = c.cpu().numpy()
            
            assert np.isnan(c).sum()==0

            np.save(os.path.join(save_dir_train, filename.replace(".png", ".npy")), c)
            
        for img, filename in tqdm(test_ds, desc=f'subj {subj} test'):
            with torch.no_grad():
                c = net.clip_encode_vision(img.unsqueeze(0))[0]
                c = c.cpu().numpy()
                
            assert np.isnan(c).sum()==0
            
            np.save(os.path.join(save_dir_test, filename.replace(".png", ".npy")), c)
