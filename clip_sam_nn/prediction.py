import argparse
import yaml
import importlib
import utils
import os
from tqdm import tqdm
import numpy as np

from dataset import Features_to_fMRI_Dataset

from models import MappingNetwork
from pl_trainer import VanillaTrainer
from loss import mse_cos_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Algonauts Challenge NN trainer")
    parser.add_argument('--config_file', type=str, default='config.yaml')
    parser.add_argument('--subj', type=int)
    parser.add_argument('--side', type=str, help='left or right side')
    parser.add_argument('--gpu', type=int, default=0) # test with only one gpu
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    utils.set_seed(args.seed)
    cfg = yaml.load(open('config.yaml', 'rb'), Loader=yaml.FullLoader)
    
    subj_idx = args.subj
    side = args.side
    seed = args.seed
    
    device = f'cuda:{args.gpu}'
    checkpoint_path = f"/SSD/slava/algonauts/clip_sam_nn_training/fmri_mapping_subj0{subj_idx}_{side}_seed_{seed}/"
    checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
    print('Checkpoint path', checkpoint_path)
    
    full_dataset = Features_to_fMRI_Dataset(subj_idx=subj_idx, side=side, mode='test')
    
    # Define model
    model = MappingNetwork(
        clip_dim=512,
        sam_dim=1024,
        out_dim=full_dataset.feat_dim
    )

    # Optimizers, etc.
    module = importlib.import_module(cfg["OPTIMIZER"]["MODULE"])
    optimizer = getattr(module, cfg["OPTIMIZER"]["CLASS"])(
        model.parameters(), **cfg["OPTIMIZER"]["ARGS"]
    )

    module = importlib.import_module(cfg["SCHEDULER"]["MODULE"])
    scheduler = getattr(module, cfg["SCHEDULER"]["CLASS"])(
        optimizer, **cfg["SCHEDULER"]["ARGS"])

    criterion = mse_cos_loss()

    cfg['SEED'] = args.seed
    cfg['OUTPUT_DIM'] = full_dataset.feat_dim

    lightning_model = VanillaTrainer.load_from_checkpoint(checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            config=cfg
    ).to(device).eval()
    
    predictions = []

    for idx, (clip_img_feat, clip_txt_feat, sam_feat, filename) in tqdm(enumerate(full_dataset)):

        # to ensure the order
        assert idx+1 == int(filename.split('_')[0].split('-')[-1])

        clip_img_feat = clip_img_feat.to(device).unsqueeze(0)
        clip_txt_feat = clip_txt_feat.to(device).unsqueeze(0)
        sam_feat = sam_feat.to(device).unsqueeze(0)

        pred = lightning_model(clip_img_feat, clip_txt_feat, sam_feat)

        pred = pred.squeeze().cpu().detach().numpy()
        
        assert not np.isnan(pred).any(), 'there is nan values'

        predictions.append(pred)

    predictions = np.array(predictions)
    
    print('Predictions shape ', predictions.shape)
    
    submission_dir = 'clip_sam_nn_submission'
    save_name = f'/SSD/slava/algonauts/{submission_dir}/'
    os.makedirs(save_name, exist_ok=True)
    save_name = os.path.join(save_name, f'subj0{subj_idx}')
    os.makedirs(save_name, exist_ok=True)
    
    if side=='right':
        save_name = os.path.join(save_name, 'rh_pred_test.npy')
    elif side=='left':
        save_name = os.path.join(save_name, 'lh_pred_test.npy')
    else:
        raise NameError

    print('Saving into ', save_name)
    np.save(save_name, predictions)
    print(f'{subj_idx} {side} is done!')