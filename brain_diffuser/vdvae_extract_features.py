import sys
sys.path.append('vdvae')

import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from tqdm import tqdm

from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

H = {'image_size': 64, 
     'image_channels': 3,
     'seed': 0, 
     'port': 29500, 
     'save_dir': './saved_models/test', 
     'data_root': './', 
     'desc': 'test', 
     'hparam_sets': 'imagenet64', 
     'restore_path': 'imagenet64-iter-1600000-model.th', 
     'restore_ema_path': 'model/imagenet64-iter-1600000-model-ema.th', 
     'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 
     'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 
     'dataset': 'imagenet64', 
     'ema_rate': 0.999, 
     'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 
     'zdim': 16, 
     'width': 512, 
     'custom_width_str': '', 
     'bottleneck_multiple': 0.25, 
     'no_bias_above': 64, 
     'scale_encblock': False, 
     'test_eval': True, 
     'warmup_iters': 100, 
     'num_mixtures': 10, 
     'grad_clip': 220.0, 
     'skip_threshold': 380.0, 
     'lr': 0.00015, 
     'lr_prior': 0.00015, 
     'wd': 0.01, 
     'wd_prior': 0.0, 
     'num_epochs': 10000, 
     'n_batch': 4, 
     'adam_beta1': 0.9, 
     'adam_beta2': 0.9, 
     'temperature': 1.0, 
     'iters_per_ckpt': 25000, 
     'iters_per_print': 1000, 
     'iters_per_save': 10000, 
     'iters_per_images': 10000, 
     'epochs_per_eval': 1, 
     'epochs_per_probe': None, 
     'epochs_per_eval_save': 1, 
     'num_images_visualize': 8, 
     'num_variables_visualize': 6, 
     'num_temperatures_visualize': 3, 
     'mpi_size': 1, 
     'local_rank': 0, 
     'rank': 0, 
     'logdir': './saved_models/test/log'}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)

class batch_generator_external_images(Dataset):

    def __init__(self, data_path, filenames):
        self.data_path = data_path
        self.filenames = filenames


    def __getitem__(self,idx):
        path = os.path.join(self.data_path, self.filenames[idx])
        img = Image.open(path)
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        return img, self.filenames[idx]

    def __len__(self):
        return  len(self.filenames)

data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
save_dir = '/SSD/slava/algonauts/brain_diffuser_vdvae_features/'

num_latents = 31

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
        
        for img, filename in tqdm(train_ds, desc=f'subj {subj} train'):
            data_input, target = preprocess_fn(img.unsqueeze(0))
            with torch.no_grad():
                activations = ema_vae.encoder.forward(data_input)
                px_z, _ = ema_vae.decoder.forward(activations, get_latents=True)
                sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
                im = sample_from_latent[0]                
            
            im = Image.fromarray(im)
            im = im.resize((512,512),resample=3)
            im.save(os.path.join(save_dir_train, filename))
            
            
        for img, filename in tqdm(test_ds, desc=f'subj {subj} test'):
            data_input, target = preprocess_fn(img.unsqueeze(0))
            with torch.no_grad():
                activations = ema_vae.encoder.forward(data_input)
                px_z, _ = ema_vae.decoder.forward(activations, get_latents=True)
                sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
                im = sample_from_latent[0]                
            
            im = Image.fromarray(im)
            im = im.resize((512,512),resample=3)
            im.save(os.path.join(save_dir_test, filename))
        
            