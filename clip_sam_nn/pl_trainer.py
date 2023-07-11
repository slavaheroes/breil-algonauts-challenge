import pytorch_lightning as pl
import torch
from torchmetrics.regression import PearsonCorrCoef

class VanillaTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_fn,
                 config) -> None:
        super(VanillaTrainer, self).__init__()
        pl.seed_everything(config["SEED"])
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
        self.corr_metric = PearsonCorrCoef(num_outputs=config["OUTPUT_DIM"])
        
        self.config = config

        self.validation_step_outputs = []
        self.training_step_outputs = []
    
    def forward(self, clip_img, clip_txt, sam_feat):
        self.model(clip_img, clip_txt, sam_feat)
    
    def calc_loss(self, x_hat, y):
        return self.loss_fn(x_hat, y)
    
    def calc_corr(self, x_hat, y):
        score = self.corr_metric(x_hat, y)
        return score.mean()

    def training_step(self, batch, batch_idx):
        clip_img, clip_txt, sam_feat, labels = batch
        outputs = self.model(clip_img, clip_txt, sam_feat)
        loss = self.calc_loss(outputs, labels)
        corr_score = self.calc_corr(outputs, labels)
        
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_pearson_corr', corr_score, on_epoch=True)
        return {'loss': loss, 'train_pearson_corr': corr_score}
    
    def validation_step(self, batch, batch_idx):
        clip_img, clip_txt, sam_feat, labels = batch
        outputs = self.model(clip_img, clip_txt, sam_feat)
        loss = self.calc_loss(outputs, labels)
        corr_score = self.calc_corr(outputs, labels)
        
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_pearson_corr', corr_score, on_epoch=True)
            
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]