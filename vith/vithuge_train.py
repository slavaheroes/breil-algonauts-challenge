# https://github.com/openai/CLIP
import os
from tqdm import tqdm
import timm
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import Trainer, seed_everything

class argObj:
  def __init__(self, data_dir, parent_submission_dir, subj):
    
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
    self.parent_submission_dir = parent_submission_dir
    self.subject_submission_dir = os.path.join(self.parent_submission_dir,
        'subj'+self.subj)

    # Create the submission directory if not existing
    if not os.path.isdir(self.subject_submission_dir):
        os.makedirs(self.subject_submission_dir)

class ViTHugeAlgonauts(nn.Module):
    def __init__(self, out=19004):
        super().__init__()

        self.vith = timm.create_model('vit_huge_patch14_224', pretrained=True)
        self.vith.head = nn.Identity()

        self.fc = nn.Sequential(
                nn.Linear(1280, 4096),
                nn.ReLU(),
                nn.Linear(4096, 19004)
        )

    def forward(self, x):
        x = self.vith(x)
        x = self.fc(x)

        return x

transform = transforms.Compose([
    transforms.Resize((224,224)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img

data_dir = '/SSD/qasymjomart/algonauts/data'
save_dir = '/SSD/qasymjomart/algonauts/breil-algonauts-challenge/vith'
parent_submission_dir = '/SSD/qasymjomart/algonauts/breil-algonauts-challenge/vith/submission'

if __name__ == "__main__":
    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--bs', type=int, help='Batch size')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    args = parser.parse_args()

    # Set up GPU devices to use
    if args.devices:
        print(f'Using GPU {args.devices}')
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
        device = f'cuda:{args.devices}'
    else:
        device = 'cpu'
    device = torch.device(device)
    batch_size = args.bs
    
    for subj in range(1, 9):

        '''
        Loading fMRI data
        '''
        fmri_dir = os.path.join(data_dir, 'training_split', 'training_fmri')
        print(fmri_dir)
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

        print('LH training fMRI data shape:')
        print(lh_fmri.shape)
        print('(Training stimulus images × LH vertices)')
        print('\nRH training fMRI data shape:')
        print(rh_fmri.shape)
        print('(Training stimulus images × RH vertices)')

        '''
        Loading images
        '''
        subj_dir = os.path.join(data_dir, 'subj'+format(subj, '02'))
        train_imgs_path = os.path.join(subj_dir, 'training_split', 'training_images')
        train_imgs = os.listdir(train_imgs_path)
        train_imgs.sort()
        print(f'Training images: {len(train_imgs)}')
        num_train = int(np.round(len(train_imgs) / 100 * 80)) # 80-20% train-val split
        idxs = np.arange(len(train_imgs))
        np.random.shuffle(idxs) # Shuffle all training stimulus images
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
        print(f'Training stimulus images: {len(idxs_train)}')
        print(f'\nValidation stimulus images: {len(idxs_val)}')
        train_imgs_paths = sorted(list(Path(train_imgs_path).iterdir()))

        train_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_train, transform), 
            batch_size=batch_size
        )
        val_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_val, transform), 
            batch_size=batch_size
        )
        
        
        '''
        Creating directories to save things
        '''
        save_dir_train = os.path.join(save_dir, 'subj'+format(subj, '02'), 'train_features')
        os.makedirs(save_dir_train, exist_ok=True)
        save_dir_test = os.path.join(save_dir, 'subj'+format(subj, '02'), 'test_features')
        os.makedirs(save_dir_test, exist_ok=True)


        ''' 
        Model init
        '''
        model = ViTHugeAlgonauts(out=19004)


        # Split fmri data
        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_val = rh_fmri[idxs_val]

        del lh_fmri, rh_fmri

        
        