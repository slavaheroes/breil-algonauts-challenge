import os
from glob import glob
import argparse

import pickle
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
submission_dir = '/SSD/slava/algonauts/stable_diffusion_v1_submission/'

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_features(paths, latent):
    features = []
    for path in paths:
        feat = load_pickle(path)[latent]
        features.append(feat.flatten())
    features = np.array(features)
    return features
        

def main(args):
    subject_submission_dir = os.path.join(submission_dir, f'latent_{args.latent}', f'subj0{args.subj}')
    os.makedirs(subject_submission_dir, exist_ok=True)
    
    print("Regression for subject ", args.subj)
    train_paths = sorted(
        glob(f"/SSD/slava/algonauts/stable_diffusion_v1_features/subj0{args.subj}/train_features/train-*.pickle")
    )
    
    fmri_dir = os.path.join(data_dir, f'subj0{args.subj}', 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    
    assert len(train_paths)==lh_fmri.shape[0]
    assert len(train_paths)==rh_fmri.shape[0]
    
    
    test_paths = sorted(
        glob(f"/SSD/slava/algonauts/stable_diffusion_v1_features/subj0{args.subj}/train_features/test-*.pickle")
    )
    
    print("Len of train: ", len(train_paths))
    print("Len of test: ", len(test_paths))
    
    train_features = load_features(train_paths, args.latent)
    test_features = load_features(test_paths, args.latent)
    
    print("Train shape: ", train_features.shape)
    print("Test shape: ", test_features.shape)
    
    # Fit some model: Currently Ridge Linear Regression
    # Do Grid Search
    parameters = {  
                'alpha': [0.001, 0.1, 1.0, 5, 10, 100, 1000],
                }
    
    reg_lh = GridSearchCV(
        estimator=Ridge(),
        param_grid=parameters
    )
    reg_lh.fit(train_features, lh_fmri)
    
    print(f'Subj {args.subj} LH scores: {reg_lh.best_score_}')
    
    reg_rh = GridSearchCV(
        estimator=Ridge(),
        param_grid=parameters
    )
    
    reg_rh.fit(train_features, rh_fmri)
    
    print(f'Subj {args.subj} RH scores: {reg_rh.best_score_}')
    
    lh_fmri_test_pred = reg_lh.predict(test_features)
    rh_fmri_test_pred = reg_rh.predict(test_features)
    print(lh_fmri_test_pred.shape, rh_fmri_test_pred.shape)
    print(f'LH is nan: {np.isnan(lh_fmri_test_pred).any()}, RH is nan: {np.isnan(rh_fmri_test_pred).any()}')
    
    # Convert to float32 before saving
    lh_fmri_test_pred = np.float32(lh_fmri_test_pred)
    rh_fmri_test_pred = np.float32(rh_fmri_test_pred)
            
    np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)
    
    print("Done")
    
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Fit Ridge regression based on Stable Diffusion latents")
    parser.add_argument('--subj', type=int)
    parser.add_argument('--latent', type=str, default='z')
    args = parser.parse_args()
    
    main(args)
    
    
    