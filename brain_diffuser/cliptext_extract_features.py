import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import json
from collections import defaultdict
from tqdm import tqdm

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)

nsd_stiminfo_file = '/SSD/slava/algonauts/algonauts_2023_challenge_data/nsd_stim_info_merged.pkl'
stiminfo = pd.read_pickle(nsd_stiminfo_file)

exp_design_file = "/SSD/slava/algonauts/algonauts_2023_challenge_data/nsd_expdesign.mat"
exp_design = loadmat(exp_design_file)

annotations = "/SSD/slava/algonauts/algonauts_2023_challenge_data/annotations/captions_train2017.json"
dataset = json.load(open(annotations, 'r'))

imgIdToAnns = defaultdict(list)

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann)

annotation_val = "/SSD/slava/algonauts/algonauts_2023_challenge_data/annotations/captions_val2017.json"
dataset = dict()
dataset = json.load(open(annotation_val, 'r'))

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann)

cocoId = np.array(stiminfo['cocoId'])

data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
save_dir = '/SSD/slava/algonauts/brain_diffuser_clip_text_features/'

if __name__=="__main__":
    for subj in range(1, 9):
        subj_dir = os.path.join(data_dir, 'subj'+format(subj, '02'))
        train_imgs_paths = os.path.join(subj_dir, 'training_split', 'training_images')
        train_imgs = sorted(os.listdir(train_imgs_paths))
        
        test_imgs_paths = os.path.join(subj_dir, 'test_split', 'test_images')
        test_imgs = sorted(os.listdir(test_imgs_paths))
        
        # features savenames
        save_dir_train = os.path.join(save_dir, 'subj'+format(subj, '02'), 'train_features')
        os.makedirs(save_dir_train, exist_ok=True)
        
        save_dir_test = os.path.join(save_dir, 'subj'+format(subj, '02'), 'test_features')
        os.makedirs(save_dir_test, exist_ok=True)

        num_embed, num_features, num_test, num_train = 257, 768, len(test_imgs), len(train_imgs)

        for filename in tqdm(train_imgs, desc=f'subj {subj} train'):
            nsd_id = filename.split("-")[-1].replace(".png", "")
            coco_id = cocoId[int(nsd_id)]
            captions = imgIdToAnns[coco_id]
            captions = [x['caption'] for x in captions]

            with torch.no_grad():
                c = net.clip_encode_text(captions)
                c = c.to('cpu').numpy().mean(0)

            assert np.isnan(c).sum()==0
            np.save(os.path.join(save_dir_train, filename.replace(".png", ".npy")), c)
            
        for filename in tqdm(test_imgs, desc=f'subj {subj} test'):
            nsd_id = filename.split("-")[-1].replace(".png", "")
            coco_id = cocoId[int(nsd_id)]
            captions = imgIdToAnns[coco_id]
            captions = [x['caption'] for x in captions]

            with torch.no_grad():
                c = net.clip_encode_text(captions)
                c = c.to('cpu').numpy().mean(0)

            assert np.isnan(c).sum()==0
            np.save(os.path.join(save_dir_test, filename.replace(".png", ".npy")), c)

