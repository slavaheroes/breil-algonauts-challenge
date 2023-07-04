import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from segment_anything import SamPredictor, sam_model_registry


data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
parent_submission_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_submission_sam'

rand_seed = 5 #@param
np.random.seed(rand_seed)


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

transform = transforms.Compose([
    transforms.Resize((1024,1024)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img

batch_size = 1
device = 'cuda:2'
device = torch.device(device)

model = sam_model_registry["vit_l"](checkpoint="/SSD/slava/pretrained_models/sam_vit_l_0b3195.pth").image_encoder
# model = torch.nn.DataParallel(model)
model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
model.eval()

def extract_features(dataloader):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        with torch.no_grad():
            d = d.to(device)
            ft = model(d)
            # Flatten the features
            ft = torch.nn.functional.avg_pool2d(ft, kernel_size=64)

        ft = ft.cpu().squeeze().detach().numpy()
        features.append(ft)
    return np.vstack(features)


if __name__=="__main__":
    for subj in range(1, 9):
        args = argObj(data_dir, parent_submission_dir, subj)
        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        print(fmri_dir)
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

        print('LH training fMRI data shape:')
        print(lh_fmri.shape)
        print('(Training stimulus images × LH vertices)')
        print('\nRH training fMRI data shape:')
        print(rh_fmri.shape)
        print('(Training stimulus images × RH vertices)')

        train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
        test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

        # Create lists will all training and test image file names, sorted
        train_img_list = os.listdir(train_img_dir)
        train_img_list.sort()
        test_img_list = os.listdir(test_img_dir)
        test_img_list.sort()
        print('Training images: ' + str(len(train_img_list)))
        print('Test images: ' + str(len(test_img_list)))

        # Calculate how many stimulus images correspond to 90% of the training data
        num_train = int(np.round(len(train_img_list) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(len(train_img_list))
        np.random.shuffle(idxs)
        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
        # No need to shuffle or split the test stimulus images
        idxs_test = np.arange(len(test_img_list))

        print('Training stimulus images: ' + format(len(idxs_train)))
        print('\nValidation stimulus images: ' + format(len(idxs_val)))
        print('\nTest stimulus images: ' + format(len(idxs_test)))


        train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
        test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

        # The DataLoaders contain the ImageDataset class
        train_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_train, transform), 
            batch_size=batch_size
        )
        val_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_val, transform), 
            batch_size=batch_size
        )
        test_imgs_dataloader = DataLoader(
            ImageDataset(test_imgs_paths, idxs_test, transform), 
            batch_size=batch_size
        )

        # split fmri data
        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_val = rh_fmri[idxs_val]

        del lh_fmri, rh_fmri

        # fit pca
        features_train = extract_features(train_imgs_dataloader)
        features_val = extract_features(val_imgs_dataloader)
        features_test = extract_features(test_imgs_dataloader)

        print('\nTraining images features:')
        print(features_train.shape)
        print('(Training stimulus images × PCA features)')

        print('\nValidation images features:')
        print(features_val.shape)
        print('(Validation stimulus images × PCA features)')

        print('\nTest images features:')
        print(features_val.shape)
        print('(Test stimulus images × PCA features)')

        # del model

        # Fit linear regressions on the training data
        reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
        reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_val_pred = reg_lh.predict(features_val)
        lh_fmri_test_pred = reg_lh.predict(features_test)
        rh_fmri_val_pred = reg_rh.predict(features_val)
        rh_fmri_test_pred = reg_rh.predict(features_test)


        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

        print("Correlation on the left side: ", np.mean(lh_correlation))
        print("Correlation on the right side: ", np.mean(rh_correlation))
        # submission files
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)

        np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

        print(f'{subj} is done.')














