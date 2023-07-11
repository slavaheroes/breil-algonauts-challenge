import argparse
import yaml
import importlib
import utils

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dataset import Features_to_fMRI_Dataset

from models import MappingNetwork
from pl_trainer import VanillaTrainer
from loss import mse_cos_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Algonauts Challenge NN trainer")
    parser.add_argument('--config_file', type=str, default='config.yaml')
    parser.add_argument('--subj', type=int)
    parser.add_argument('--side', type=str, help='left or right side')
    parser.add_argument('--gpu', type=int, default=0) # train with only one gpu
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    FILENAME_POSTFIX = f'fmri_mapping_subj0{args.subj}_{args.side}_seed_{args.seed}'
    # Load config file
    cfg = yaml.load(open(args.config_file, 'rb'), Loader=yaml.FullLoader)
    cfg['SAVENAME'] = FILENAME_POSTFIX
    cfg['SUBJ'] = args.subj
    cfg['SIDE'] = args.side
    cfg['SEED'] = args.seed
    
    utils.set_seed(args.seed)
    
    logger = WandbLogger(project="BREIL-Algonauts-challenge",
                    name=FILENAME_POSTFIX,
                    save_dir="/SSD/slava/algonauts/wandb_checkpoints",
                    log_model=False
                    )
    
    generator = torch.Generator().manual_seed(args.seed)
    
    # Make data
    full_dataset = Features_to_fMRI_Dataset(subj_idx=args.subj, side=args.side, mode='train')
    train_size = 9700 #int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size], generator=generator)
    
    print("Train size: ", len(train_dataset))
    print("Valid size: ", len(valid_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'])
    
    cfg['OUTPUT_DIM'] = full_dataset.feat_dim
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
    
    # Import callbacks
    callbacks = []
    if cfg["EARLY_STOPPING"]["ENABLE"]:
        early_stop_callback = EarlyStopping(
            **cfg["EARLY_STOPPING"]["ARGS"]
        )
        callbacks.append(early_stop_callback)
    
    if cfg["LR_MONITOR"]["ENABLE"]:
        lr_monitor_callback = LearningRateMonitor(
             **cfg["LR_MONITOR"]["ARGS"]
        )
        callbacks.append(lr_monitor_callback)

    cfg["CHECKPOINT"]["ARGS"]["dirpath"] = f'/SSD/slava/algonauts/clip_sam_nn_training/{FILENAME_POSTFIX}'
    checkpoint_callback = ModelCheckpoint(
        **cfg["CHECKPOINT"]["ARGS"]
    )
    callbacks.append(checkpoint_callback)
    
    grad_clip = cfg["GRADIENT_CLIPPING"]
    grad_acum = cfg["GRADIENT_ACCUMULATION_STEPS"]
    
    lightning_model = VanillaTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=criterion,
        config=cfg
    )
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu],
        max_epochs=cfg["EPOCHS"],
        num_sanity_val_steps=0,
        # limit_train_batches=0.1,
        # limit_val_batches=0.5,
        check_val_every_n_epoch=cfg["CHECK_VAL_EPOCHS"],
        logger=logger,
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=grad_acum,
        callbacks=callbacks,
        log_every_n_steps=1,
    )
    
    # valid, valid is for one-batch overfit, then will be changed to train, valid
    trainer.fit(lightning_model, valid_loader, valid_loader)