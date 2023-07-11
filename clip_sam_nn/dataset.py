import numpy as np
import os

import torch

from torch.utils.data.dataset import Dataset


class Features_to_fMRI_Dataset(Dataset):
    def __init__(self, 
                 subj_idx,
                 side, # left or right
                 mode, # train or test
                 main_data_dir="/SSD/slava/algonauts/algonauts_2023_challenge_data/",
                 clip_img_feat_dir="/SSD/slava/algonauts/clip_base_features/",
                 clip_txt_feat_dir="/SSD/slava/algonauts/clip_text_features/",
                 sam_feat_dir="/SSD/slava/algonauts/sam_large_features/"):
        
        self.subj_idx = subj_idx
        self.side = side
        self.mode = mode
        
        if mode=='train':
            self.tail_dir = os.path.join(f'subj0{subj_idx}', 'train_features')
            self.label_path = os.path.join(
                main_data_dir, f'subj0{subj_idx}', 'training_split', 'training_fmri'
            )

            if side=='left':
                self.label_path = os.path.join(self.label_path, 'lh_training_fmri.npy')
            elif side=='right':
                self.label_path = os.path.join(self.label_path, 'rh_training_fmri.npy')
            else:
                raise NameError
            
            assert os.path.exists(self.label_path), f"Label path is wrong: {self.label_path}"

        elif mode=='test':
            self.tail_dir = os.path.join(f'subj0{subj_idx}', 'test_features')
            self.label_path = None
        
        else:
            raise NameError
        
        self.clip_img_feat_dir = os.path.join(clip_img_feat_dir, self.tail_dir)
        self.clip_txt_feat_dir = os.path.join(clip_txt_feat_dir, self.tail_dir)
        self.sam_feat_dir = os.path.join(sam_feat_dir, self.tail_dir)

        assert os.path.exists(self.clip_img_feat_dir)
        assert os.path.exists(self.clip_txt_feat_dir)
        assert os.path.exists(self.sam_feat_dir)

        self.filenames = sorted(
            os.listdir(self.clip_img_feat_dir)
        )
        print("First five filenames: ", self.filenames[:5])

        if self.label_path:
            self.labels = np.load(self.label_path)
            print("Shape of labels: ", self.labels.shape)
            self.feat_dim = self.labels.shape[1]
        else:
            print("Test mode, no labels path")
            self.feat_dim = -1
            self.labels = None

        self.num_of_cases = len(self.filenames)
        
    def __len__(self):
        return self.num_of_cases
    
    def __str__(self):
        return f'Subj {self.subj_idx}, side {self.side}, mode {self.mode}, labels dim {self.feat_dim}'
    
    def _make_tensor(self, x):
        return torch.from_numpy(x).float()
    
    def _transform_sam_feat(self, img_feat):
        img_feat = torch.nn.functional.avg_pool2d(
            torch.from_numpy(img_feat), kernel_size=32
        ).view(-1)
        return img_feat.float()
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]

        clip_img = np.load(
            os.path.join(self.clip_img_feat_dir, filename)
        )
        clip_img = self._make_tensor(clip_img)

        clip_txt = np.load(
            os.path.join(self.clip_txt_feat_dir, filename)
        )
        clip_txt = self._make_tensor(clip_txt)

        sam_feat = np.load(
            os.path.join(self.sam_feat_dir, filename)
        ) 
        sam_feat = self._transform_sam_feat(sam_feat) 
        
        if self.mode=='train':
            label = self.labels[idx, :]
            label = self._make_tensor(label)
            return clip_img, clip_txt, sam_feat, label

        return clip_img, clip_txt, sam_feat      