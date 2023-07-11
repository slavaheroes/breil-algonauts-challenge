# from https://github.com/sklin93/mind-reader/blob/main/fmri_clip_utils.py

import torch
import torch.nn as nn

class cos_sim_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, y_pred, y):
        target = torch.ones(len(y)).to(y.device)
        return self.loss_fn(y_pred, y, target)


class mse_cos_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cos_loss = cos_sim_loss()

    def forward(self, y_pred, y, cos_weight=0.5):
        return ((1 - cos_weight) * self.mse_loss(y_pred, y) +
                cos_weight * self.cos_loss(y_pred, y))


class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y, temperature=0.5, lam=0.5, cos_weight=None):
        # _dot = torch.bmm(y.view(len(y), 1, -1),
        #     y_pred.view(len(y_pred), -1, 1)).squeeze()
        # _norm = torch.norm(y, dim=1) * torch.norm(y_pred, dim=1)
        # cos_sim = _dot/ _norm # shape (32,)
        # loss = -F.log_softmax(cos_sim / temperature, dim=0)
        # return torch.mean(loss)

        sim = torch.cosine_similarity(y_pred.unsqueeze(1), y.unsqueeze(0), dim=-1)
        # sim: shape (32, 32), diagonal is equivalant to above cos_sim
        if temperature > 0.:
            sim = sim / temperature
            # the above loss = - F.log_softmax(torch.diagonal(sim), dim=0)
            # whereas the below loss = - torch.diagonal(F.log_softmax(sim, dim=0))
            sim1 = torch.diagonal(F.log_softmax(sim, dim=1))
            sim2 = torch.diagonal(F.log_softmax(sim, dim=0))
            return (-(lam * sim1 + (1. - lam) * sim2)).mean()
        else:
            return (-torch.diagonal(sim)).mean()


class mse_cos_contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_cos_loss = mse_cos_loss()
        self.contrastive_loss = contrastive_loss()
    def forward(self, y_pred, y, temperature=0.5, cos_weight=0.5, contra_p=0.5):
        return ((1 - contra_p) * self.mse_cos_loss(y_pred, y, cos_weight) +
            contra_p * self.contrastive_loss(y_pred, y, temperature))