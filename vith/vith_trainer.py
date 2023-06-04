'''

Inspired by Slavaheroes' code at https://github.com/qasymjomart/DAMIT_v1/blob/lightning-branch/lightning_learners/classification_trainer.py

'''

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class MedianNNCorr(torch.nn.Module):
    def __init__(self):
        super(MedianNNCorr, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the median noise-normalized squared correlation.

        Args:
            predictions (torch.Tensor): Predicted values of shape (batch_size, num_samples).
            targets (torch.Tensor): Target values of shape (batch_size, num_samples).

        Returns:
            torch.Tensor: Median noise-normalized squared correlation.
        """
        # Compute the squared correlation
        corr = torch.pow(F.cosine_similarity(predictions, targets, dim=1), 2)

        # Compute the noise-normalized squared correlation
        nn_corr = corr / (torch.var(predictions, dim=1) * torch.var(targets, dim=1) + 1e-8)

        # Compute the median of the noise-normalized squared correlation
        median_nn_corr = torch.median(nn_corr)

        return median_nn_corr


class AlgonautsTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 seed) -> None:
        super().__init__()
        pl.seed_everything(seed)
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = torchmetrics.PearsonCorrCoef()

        self.pearson_coeff = torchmetrics.PearsonCorrCoef()
        self.nn_corr = MedianNNCorr()

        self.validation_step_outputs = []
        self.training_step_outputs = []
    
    def forward(self, x):
        self.model(x)
    
    def calc_loss(self, x, y):
        return self.loss_fn(x, y)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = 1 - self.calc_loss(outputs, labels)
        nn_corr_coeff = self.nn_corr(outputs, labels)
        self.log('loss', loss, on_epoch=True, prog_bar=True)

        # calculate the metrics
        self.training_step_outputs.append(
            {
            "loss": loss,
            "nn_corr_coeff": nn_corr_coeff
            }
        )
        self.log("train_loss", loss)
        self.log("train_nn_corr_coeff", nn_corr_coeff)
        
        return {'loss': loss}

    def on_train_epoch_end(self):

        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = 1 - self.pearson_coeff(outputs, labels)
        nn_corr_coeff = self.nn_corr(outputs, labels)

        
        self.validation_step_outputs.append(
            {   
                'valid_loss': loss,
                "valid_nn_corr_coeff": nn_corr_coeff,
            }
        )

        return {'valid_loss': loss}
    
    def on_validation_epoch_end(self):

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]