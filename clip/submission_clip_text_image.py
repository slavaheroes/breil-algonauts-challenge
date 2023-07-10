import os
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
parent_submission_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_submission_clip_text_n_image'


def load_dataset(subj, mode):
    if mode=='train':
        mode = 'train_features'
    elif mode=='test':
        mode = 'test_features'
        
    image_features_dir = os.path.join("/SSD/slava/algonauts/clip_base_features", f'subj0{subj}', mode)
    text_features_dir = os.path.join("/SSD/slava/algonauts/clip_text_features", f'subj0{subj}', mode)
    
    
    npy_img_files = sorted(os.listdir(image_features_dir))
    npy_text_files = sorted(os.listdir(text_features_dir))
        
    image_features = []
    
    for npy_img, npy_text in zip(npy_img_files, npy_text_files):
        
        feat_1 = np.load(os.path.join(image_features_dir, npy_img))
        feat_2 = np.load(os.path.join(text_features_dir, npy_text))
        img_feat = np.concatenate((feat_1, feat_2), axis=0)
        
        assert len(img_feat)==1024, f'{npy_img} {feat_1.shape} {npy_text} {feat_2.shape}'
        image_features.append(img_feat)
            
    image_features = np.array(image_features)
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
        print("Start of ", subj)
        
        train_features = load_dataset(subj, 'train')
        test_features = load_dataset(subj, 'test')
        
        print("Train feats: ", train_features.shape)
        print("Test feats: ", test_features.shape)
            
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
        print(lh_fmri_test_pred.shape, rh_fmri_test_pred.shape)
        print(f'LH: {np.isnan(lh_fmri_test_pred).any()}, RH: {np.isnan(rh_fmri_test_pred).any()}')
        
        # Convert to float32 before saving
        lh_fmri_test_pred = np.float32(lh_fmri_test_pred)
        rh_fmri_test_pred = np.float32(rh_fmri_test_pred)
                
        np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

        print(f'{subj} is done.')
              