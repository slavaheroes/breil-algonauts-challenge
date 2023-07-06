import os
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import torch

features_dir = '/SSD/slava/algonauts/sam_large_features'
data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
parent_submission_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_submission_sam_extended'


def load_dataset(npy_dir):
    npy_files = sorted(os.listdir(npy_dir))    
    image_features = []
    
    for npy_file in tqdm(npy_files, desc='feature_extraction'):
        img_feat = np.load(os.path.join(npy_dir, npy_file))
        
        img_feat = torch.nn.functional.avg_pool2d(
            torch.Tensor(img_feat), kernel_size=32
        ).view(-1).numpy()
        
        image_features.append(img_feat)
    
    image_features = np.array(image_features)
    print(image_features.shape)
    return image_features
        


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

if __name__ == "__main__":
        
    for subj in range(1, 9):
        print("Start of subject ", subj)
        
        train_path = os.path.join(features_dir, 'subj'+format(subj, '02'), 'train_features')
        test_path = os.path.join(features_dir, 'subj'+format(subj, '02'), 'test_features')
        train_features = load_dataset(train_path)
        test_features = load_dataset(test_path)
        
        args = argObj(data_dir, parent_submission_dir, subj)
        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')

        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
            
        # Fit some model: Currently Ridge Linear Regression
        # Do Grid Search
        parameters = {  
                    'alpha': [0.1, 1.0, 5, 10, 100],
                    }
        
        reg_lh = GridSearchCV(
            estimator=Ridge(),
            param_grid=parameters
        )
        reg_lh.fit(train_features, lh_fmri)
        
        print(f'Subj {subj} LH scores: {reg_lh.best_score_}')
        
        reg_rh = GridSearchCV(
            estimator=Ridge(),
            param_grid=parameters
        )
        
        reg_rh.fit(train_features, rh_fmri)
        
        print(f'Subj {subj} RH scores: {reg_rh.best_score_}')
        
        lh_fmri_test_pred = reg_lh.predict(test_features)
        rh_fmri_test_pred = reg_rh.predict(test_features)
        print(f'LH: {np.isnan(lh_fmri_test_pred).any()}, RH: {np.isnan(rh_fmri_test_pred).any()}')
                
        np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

        print(f'{subj} is done.')
              