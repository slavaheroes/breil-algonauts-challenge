import pandas as pd
import json
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
import os
from tqdm import tqdm

from llama_cpp import Llama
llm = Llama(model_path="/SSD/slava/llama.cpp/models/llama-2-70b/ggml-model-q4_0.gguf", embedding=True)

nsd_stiminfo_file = '/SSD/slava/algonauts/algonauts_2023_challenge_data/nsd_stim_info_merged.pkl'
stiminfo = pd.read_pickle(nsd_stiminfo_file)

exp_design_file = "/SSD/slava/algonauts/algonauts_2023_challenge_data/nsd_expdesign.mat"
exp_design = loadmat(exp_design_file)

subject_idx  = exp_design['subjectim']

cocoId_arr = np.zeros(shape=(8, 100000), dtype=int)

for j in range(len(subject_idx)):
    cocoId = np.array(stiminfo['cocoId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
    nsdId = np.array(stiminfo['nsdId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
    assert cocoId.shape == nsdId.shape
    for i in range(nsdId.shape[0]):
        cocoId_arr[j, nsdId[i]] = cocoId[i]
    
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
        
save_dir = "/SSD/slava/algonauts/llama-2-70b_q4_0-embeddings/"
cocoId = np.array(stiminfo['cocoId'])

for subj_idx in range(1, 9):
    img_dir_train = f'/SSD/slava/algonauts/algonauts_2023_challenge_data/subj0{subj_idx}/training_split/training_images/'
    img_dir_test = f'/SSD/slava/algonauts/algonauts_2023_challenge_data/subj0{subj_idx}/test_split/test_images/'

    for img_dir in [img_dir_train, img_dir_test]:

        subj_save_dir = os.path.join(save_dir, 'subj'+format(subj_idx, '02'))
        os.makedirs(subj_save_dir, exist_ok=True)

        if 'train' in img_dir:
            feat_save_dir = os.path.join(subj_save_dir, 'train_features')
        else:
            feat_save_dir = os.path.join(subj_save_dir, 'test_features')

        os.makedirs(feat_save_dir, exist_ok=True)
        
        for img_name in tqdm(sorted(os.listdir(img_dir)), desc=img_dir):
            img_path = os.path.join(img_dir, img_name)
        
            nsd_id = img_name.split("-")[-1].replace(".png", "")
            coco_id = cocoId[int(nsd_id)]
            
            captions = imgIdToAnns[coco_id]
            captions = [x['caption'] for x in captions]
    
            feats = []
            for text in captions:
                feats.append(
                    llm.embed(text)
                )
            
            feats = np.array(feats)
            
            with open(
                os.path.join(feat_save_dir, img_name.replace(".png", ".npy")), 'wb'
            ) as f:
                np.save(f, feats)

    break