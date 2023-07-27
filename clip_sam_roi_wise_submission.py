import os
import argparse
import time

import numpy as np
import torch
import einops

from sklearn.linear_model import RidgeCV

from utils import *

import warnings
warnings.filterwarnings("ignore")

data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
parent_submission_dir = '/SSD/slava/algonauts/last_submissions/clip_base_vision_text_sam_feats_per_each_roi_submission'


def load_dataset(subj, mode):
    if mode=='train':
        mode = 'train_features'
    elif mode=='test':
        mode = 'test_features'
        
    image_features_dir = os.path.join("/SSD/slava/algonauts/clip_base_features", f'subj0{subj}', mode)
    text_features_dir = os.path.join("/SSD/slava/algonauts/clip_text_features", f'subj0{subj}', mode)
    sam_features_dir = os.path.join('/SSD/slava/algonauts/sam_large_features', f'subj0{subj}', mode)
    
    
    npy_img_files = sorted(os.listdir(image_features_dir))
    npy_text_files = sorted(os.listdir(text_features_dir))
    npy_sam_files = sorted(os.listdir(sam_features_dir))
    
    
    clip_vision_feats = []
    clip_vision_text_feats = []
    sam_feats = []
    image_features = []
    
    for npy_img, npy_text, npy_sam in zip(npy_img_files, npy_text_files, npy_sam_files):
        
        feat_1 = np.load(os.path.join(image_features_dir, npy_img))
        feat_2 = np.load(os.path.join(text_features_dir, npy_text))
        
        # add array of clip feats
        clip_vision_feats.append(feat_1)
        clip_vision_text_feats.append(
            np.concatenate((feat_1, feat_2), axis=0) 
        )
        
        # sam features with 16 kernel
        feat_3 = np.load(os.path.join(sam_features_dir, npy_sam))
        feat_3 = torch.nn.functional.avg_pool2d(
            torch.Tensor(feat_3), kernel_size=16
        ).view(-1).numpy()
        
        sam_feats.append(feat_3)
        
        img_feat = np.concatenate((feat_1, feat_2, feat_3), axis=0)        
        image_features.append(img_feat)
    
    clip_vision_feats = np.array(clip_vision_feats)
    clip_vision_text_feats = np.array(clip_vision_text_feats)
    sam_feats = np.array(sam_feats)
            
    image_features = np.array(image_features)
    
    return clip_vision_feats, clip_vision_text_feats, sam_feats, image_features
        


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
    parser = argparse.ArgumentParser(description="Algonauts Challenge per ROI submission")
    parser.add_argument('--subj', type=int, default=1)
    args = parser.parse_args()
    subj = args.subj
    
    print("Start of subj", subj)
    
    train_clip_vision_feats, train_clip_vision_text_feats, train_sam_feats, train_image_features = load_dataset(subj, 'train')
    test_clip_vision_feats, test_clip_vision_text_feats, test_sam_feats, test_image_features = load_dataset(subj, 'test')
    
    print("Train features ", train_image_features.shape, "Max ", train_image_features.max(), "Min ", train_image_features.min())
    print("Test features ", test_image_features.shape, "Max ", test_image_features.max(), "Min ", test_image_features.min())

    args = argObj(data_dir, parent_submission_dir, subj)
    
    # Load fMRI
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')

    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))        
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    
    print("LH fMRI ", lh_fmri.shape, "Max ", lh_fmri.max(), "Min ", lh_fmri.min())
    print("RH fMRI ", rh_fmri.shape, "Max ", rh_fmri.max(), "Min ", rh_fmri.min())
    
    # hyperparameter space
    # alpha_values = (0.01, 0.1, 1.0, 100, 1000, 10000)
    alpha_values = (1.0, 1000)
    print(alpha_values)
    
    for hemisphere in ['r', 'l']:
        if os.path.exists(
            os.path.join(args.subject_submission_dir, f'{hemisphere}h_pred_test.npy')
        ):
            continue
        
        if hemisphere=='r':
            label_arr = rh_fmri
        else:
            label_arr = lh_fmri
            
        pred_array = np.zeros((test_image_features.shape[0], label_arr.shape[1]))
        roi_zones_array = np.zeros((label_arr.shape[1], ))
        
        # count = 0
        
        for roi, best_id in best_ids_dict.items():
            roi_class = get_roi_class(roi)
            
            # Load ROI brain surface maps
            challenge_roi_class_dir = os.path.join(data_dir, f'subj0{subj}', 'roi_masks', hemisphere+'h.'+roi_class+'_challenge_space.npy')
            roi_map_dir = os.path.join(data_dir, f'subj0{subj}', 'roi_masks', 'mapping_'+roi_class+'.npy')
            challenge_roi_class = np.load(challenge_roi_class_dir)
            roi_map = np.load(roi_map_dir, allow_pickle=True).item()
            
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
            challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)

            # remove intersections
            challenge_roi = challenge_roi - (challenge_roi*roi_zones_array)
            
            # empty roi zone
            if challenge_roi.sum()==0:
                continue
            
            
            assert challenge_roi.max()==1
            assert challenge_roi.min()==0
            
            roi_zones_array = roi_zones_array + challenge_roi
                    
            challenge_roi_train = einops.repeat(challenge_roi, 'h -> n h', n=train_image_features.shape[0])
            challenge_roi_test = einops.repeat(challenge_roi, 'h -> n h', n=test_image_features.shape[0])
            
            labels = label_arr*challenge_roi_train
            
            assert (labels!=0).sum() == challenge_roi_train.sum(), f'Subj {subj}, ROI {hemisphere + "_" + roi} {(labels!=0).sum()} {challenge_roi_train.sum()}'
            
            # count += challenge_roi_train.sum()
            
            if best_id==0:
                suitable_features = train_clip_vision_feats
                test_features = test_clip_vision_feats
            elif best_id==1:
                suitable_features = train_clip_vision_text_feats
                test_features = test_clip_vision_text_feats
            elif best_id==2:
                suitable_features = train_sam_feats
                test_features = test_sam_feats
            
            # Fit Ridge Regression with CV grid search
            print(f'Start of ROI {hemisphere, roi} prediction. Features id: {best_id}, Train shape: {suitable_features.shape}, Test shape: {test_features.shape}')
            
            start = time.time()
            
            a_model = RidgeCV(alphas=alpha_values)
            a_model.fit(suitable_features, labels)
            roi_pred = a_model.predict(test_features)
            
            end = time.time()
            print(f'ROI {hemisphere, roi_class} pred is finished in {end-start :.3f}, best alpha: {a_model.alpha_}, best_score: {a_model.best_score_}')
            print(f'Pred shape: {roi_pred.shape}, Non-zero: {(roi_pred!=0).sum(), challenge_roi_test.sum()}')
            
            # roi_pred = roi_pred*challenge_roi_test
            
            pred_array = pred_array + roi_pred
                    
        assert roi_zones_array.max()==1
        assert roi_zones_array.min()==0
        
        non_roi_zones = (1 - roi_zones_array)
        
        non_roi_train = einops.repeat(non_roi_zones, 'h -> n h', n=train_image_features.shape[0])
        non_roi_test = einops.repeat(non_roi_zones, 'h -> n h', n=test_image_features.shape[0])
        
        non_roi_labels = non_roi_train * label_arr
        
        print("Start of non-ROI regression: ", hemisphere, (non_roi_labels!=0).sum(), non_roi_train.sum())
        
        # count += non_roi_train.sum()
        
        suitable_features = train_image_features
                
        start = time.time()
        a_model = RidgeCV(alphas=alpha_values)
        a_model.fit(suitable_features, non_roi_labels)
        non_roi_preds = a_model.predict(test_image_features)
        end = time.time()
        
        print(f'Non ROI {hemisphere} pred is finished in {end-start :.3f}, best_alpha: {a_model.alpha_}, best_score: {a_model.best_score_}')
        print(f'Pred shape: {non_roi_preds.shape}, Non-zero: {(non_roi_preds!=0).sum(), non_roi_test.sum()}')
        
        # non_roi_preds = non_roi_preds*non_roi_test
        
        pred_array = pred_array + non_roi_preds
        
        print(f'Pred for {hemisphere} is finished. Max: {pred_array.max()}, Min: {pred_array.min()}')
        
        # save
        pred_array = np.float32(pred_array)
        np.save(os.path.join(args.subject_submission_dir, f'{hemisphere}h_pred_test.npy'), pred_array)
        
        # just for some test
        # print(count, train_image_features.shape[0]*train_image_features.shape[1])
        
    print(f'{subj} is done.')