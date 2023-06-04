'''

Inspired by Slavaheroes's code at https://github.com/qasymjomart/DAMIT_v1/blob/lightning-branch/conv_nets_training.py

'''

import os
from tqdm import tqdm
import timm
import argparse
import yaml
from pathlib import Path

import numpy as np
from PIL import Image
import random
import wandb

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# from lightning.pytorch import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from vith_trainer import AlgonautsTrainer

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

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

class AlgonautsImageDataset(Dataset):
    def __init__(self, imgs_paths, lh_fmri, rh_fmri, idxs, transform, hemisphere='left'):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.lh_fmri = lh_fmri
        self.rh_fmri = rh_fmri
        self.transform = transform
        self.hemisphere = hemisphere

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        lh_fmri_value = self.lh_fmri[idx]
        rh_fmri_value = self.rh_fmri[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        if self.hemisphere == 'left':
            fmri_to_return = torch.tensor(lh_fmri_value).cuda()
        elif self.hemisphere == 'right':
            fmri_to_return = torch.tensor(rh_fmri_value).cuda()
        elif self.hemisphere == 'both':
            fmri_to_return = torch.tensor(np.concatenate((lh_fmri_value, rh_fmri_value))).cuda()
        return img, fmri_to_return

data_dir = '/SSD/qasymjomart/algonauts/data'
save_dir = '/SSD/qasymjomart/algonauts/breil-algonauts-challenge/vith'
parent_submission_dir = '/SSD/qasymjomart/algonauts/breil-algonauts-challenge/vith/submission'

if __name__ == "__main__":
    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    args = parser.parse_args()

    EPOCHS = args.epochs
    SEED = args.seeds
    set_seed(SEED)

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
        Logger init
        '''
        logger = WandbLogger(project='Algonauts Training',
                             name = f'ViT-H training of Subject {subj}',
                             save_dir=f'{save_dir}/logs',
                             log_model=False
                             )
        '''
        Loading fMRI data
        '''
        subj_dir = os.path.join(data_dir, 'subj'+format(subj, '02'))
        fmri_dir = os.path.join(subj_dir, 'training_split', 'training_fmri')
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
        
        '''
        Creating directories to save things
        '''
        save_dir_train = os.path.join(save_dir, 'subj'+format(subj, '02'), 'train_features')
        os.makedirs(save_dir_train, exist_ok=True)
        save_dir_test = os.path.join(save_dir, 'subj'+format(subj, '02'), 'test_features')
        os.makedirs(save_dir_test, exist_ok=True)


        ''' 
        Model init, optimizer, scheduler init
        '''
        model = ViTHugeAlgonauts(out=19004)
        optimizer = torch.optim.AdamW(lr = 0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=EPOCHS, eta_min=0.0000001)


        ''' 
        Split fmri data
        '''
        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_val = rh_fmri[idxs_val]

        del lh_fmri, rh_fmri

        train_dataloader = DataLoader(
            AlgonautsImageDataset(train_imgs_paths, lh_fmri_train, rh_fmri_train, idxs_train, transform, hemisphere='left'), 
            batch_size=batch_size
        )
        val_datalaoder = DataLoader(
            AlgonautsImageDataset(train_imgs_paths, lh_fmri_val, rh_fmri_val, idxs_val, transform, hemisphere='left'), 
            batch_size=batch_size
        )

        '''
        Training callback
        '''
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
                save_dir
            )
        callbacks.append(checkpoint_callback)

        '''
        Training class
        '''
        # Init lightning model and .fit()
        ligtning_model = AlgonautsTrainer(
            model,
            optimizer,
            scheduler,
            SEED
        )

        grad_clip = None
        grad_acum = 1
        trainer = pl.Trainer(
                accelerator="gpu",
                devices=[0],
                max_epochs=EPOCHS,
                num_sanity_val_steps=0,
                # limit_train_batches=0.05,
                # limit_val_batches=0.25,
                check_val_every_n_epoch = 1,
                logger=logger,
                gradient_clip_val=grad_clip,
                accumulate_grad_batches=grad_acum,
                callbacks=callbacks,
                log_every_n_steps=1,
            )

        trainer.fit(ligtning_model, train_dataloader, val_datalaoder)

        wandb.finish()


        
        