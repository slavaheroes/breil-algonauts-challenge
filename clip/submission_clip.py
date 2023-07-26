import os
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

features_dir = '/SSD/slava/algonauts/clip_text_features'
data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
parent_submission_dir = '/SSD/slava/algonauts/last_submissions/clip_base_text_only_without_normalization'


def load_dataset(npy_dir):
    npy_files = sorted(os.listdir(npy_dir))    
    image_features = []
    
    for npy_file in npy_files:
        img_feat = np.load(os.path.join(npy_dir, npy_file))
        
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
        print("Start of subj", subj)
        train_path = os.path.join(features_dir, 'subj'+format(subj, '02'), 'train_features')
        test_path = os.path.join(features_dir, 'subj'+format(subj, '02'), 'test_features')
        train_features = load_dataset(train_path)
        test_features = load_dataset(test_path)
        
        print("Train features ", train_features.shape, "Max ", train_features.max(), "Min ", train_features.min())
        print("Test features ", test_features.shape, "Max ", test_features.max(), "Min ", test_features.min())

        args = argObj(data_dir, parent_submission_dir, subj)
        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')

        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        print("Original LH fMRI ", lh_fmri.max(), lh_fmri.min())
        
        # z-normalization
        # lh_fmri_mean, lh_fmri_std = lh_fmri.mean(axis=0), lh_fmri.std(axis=0, ddof=1)
        # lh_fmri = (lh_fmri-lh_fmri_mean)/lh_fmri_std
        
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
        print("Original RH fMRI ", rh_fmri.max(), rh_fmri.min())
        
        # z-normalization
        # rh_fmri_mean, rh_fmri_std = rh_fmri.mean(axis=0), rh_fmri.std(axis=0, ddof=1)
        # rh_fmri = (rh_fmri-rh_fmri_mean)/rh_fmri_std
        
        print("LH fMRI ", lh_fmri.shape, "Max ", lh_fmri.max(), "Min ", lh_fmri.min())
        print("RH fMRI ", rh_fmri.shape, "Max ", rh_fmri.max(), "Min ", rh_fmri.min())
        # Fit some model: Currently Ridge Linear Regression
        # Do Grid Search
        alpha_values = (0.000001,0.00001,0.0001, 0.01, 0.1, 1.0, 100, 1000, 10000, 100000)
        
        preprocess_pipe = make_pipeline(
           StandardScaler(with_mean=True, with_std=True)
        )

        reg_lh = make_pipeline(
        #    preprocess_pipe,
           RidgeCV(alphas=alpha_values)
        )
        reg_lh.fit(train_features, lh_fmri)
        
        print(f'Subj {subj} LH scores: {reg_lh[0].best_score_}')
        print(f'Subj {subj} LH best alpha: {reg_lh[0].alpha_}')
        
        reg_rh = make_pipeline(
        #    preprocess_pipe,
           RidgeCV(alphas=alpha_values)
        )
        
        reg_rh.fit(train_features, rh_fmri)
        
        print(f'Subj {subj} RH scores: {reg_rh[0].best_score_}')
        print(f'Subj {subj} RH best alpha: {reg_rh[0].alpha_}')
        
        lh_fmri_test_pred = reg_lh.predict(test_features)
        rh_fmri_test_pred = reg_rh.predict(test_features)
        
        # z denormalization
        # lh_fmri_test_pred = (lh_fmri_test_pred-lh_fmri_test_pred.mean(axis=0))/lh_fmri_test_pred.std(axis=0)
        # rh_fmri_test_pred = (rh_fmri_test_pred-rh_fmri_test_pred.mean(axis=0))/rh_fmri_test_pred.std(axis=0)
        
        # lh_fmri_test_pred = lh_fmri_test_pred*lh_fmri_std + lh_fmri_mean
        # rh_fmri_test_pred = rh_fmri_test_pred*rh_fmri_std + rh_fmri_mean
        
        print("LH fMRI pred", lh_fmri_test_pred.shape, "Max ", lh_fmri_test_pred.max(), "Min ", lh_fmri_test_pred.min())
        print("RH fMRI pred", rh_fmri_test_pred.shape, "Max ", rh_fmri_test_pred.max(), "Min ", rh_fmri_test_pred.min())
        
        print(f'LH: {np.isnan(lh_fmri_test_pred).any()}, RH: {np.isnan(rh_fmri_test_pred).any()}')
        
        # exit()
        # Convert to float32 before saving
        lh_fmri_test_pred = np.float32(lh_fmri_test_pred)
        rh_fmri_test_pred = np.float32(rh_fmri_test_pred)
        
        print("Save dir ", args.subject_submission_dir)

        np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

        print(f'{subj} is done.')