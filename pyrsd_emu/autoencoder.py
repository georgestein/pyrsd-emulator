import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from functools import partial

class AutoEncoder(pl.LightningModule):
    """

    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.train_predictor = self.hparams.get('train_predictor', False)
        self.parameter_dim = self.hparams.get('parameter_dim', 11)

        self.input_dim = self.hparams.get('input_dim', 100)
        self.hidden_dims = self.hparams.get('hidden_dims', [8,4])
        self.hidden_dims_predict = self.hparams.get('hidden_dims_predict', [8,4])
        self.batch_norm = self.hparams.get('batch_norm', False)

        self.latent_dim = self.hparams.get('latent_dim', 1)
        self.learning_rate = self.hparams.get('learning_rate', 1e-1)
        self.optimizer = self.hparams.get('optimizer', 'Adam')
        self.scheduler = self.hparams.get('scheduler', 'CosineAnnealingLR')
        self.T_max = self.hparams.get('T_max', 25)

        self.scheduler_params = self.hparams.get(
            'scheduler_params',
            {
                'T_max': self.T_max,
                'eta_min': 0,
             }
        )
        
        self.make_encoder()
        self.make_decoder()
        self.make_predictor()
                 
    def make_encoder(self):
        encoder = []
        encoder.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        encoder.append(nn.ReLU())
        if self.batch_norm:
            encoder.append(nn.BatchNorm1d(self.hidden_dims[0]))
              
        for i in range(len(self.hidden_dims) - 1):
            encoder.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            encoder.append(nn.ReLU())
            if self.batch_norm:
                encoder.append(nn.BatchNorm1d(self.hidden_dims[i+1]))
              
        encoder.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
        
        self.encoder = nn.Sequential(*encoder)
 
    def make_decoder(self):
        decoder = []
        decoder.append(nn.Linear(self.latent_dim, self.hidden_dims[-1]))
        decoder.append(nn.ReLU())
        if self.batch_norm:
            decoder.append(nn.BatchNorm1d(self.hidden_dims[-1]))
              
        for i in range(len(self.hidden_dims)-1, 0, -1):
            print(i)
            decoder.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i-1]))
            decoder.append(nn.ReLU())
            if self.batch_norm:
                decoder.append(nn.BatchNorm1d(self.hidden_dims[i-1]))
             
        decoder.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        
        self.decoder = nn.Sequential(*decoder)

    def make_predictor(self):
        predictor = []
        predictor.append(nn.Linear(self.latent_dim, self.hidden_dims_predict[0]))
        predictor.append(nn.ReLU())
        if self.batch_norm:
            predictor.append(nn.BatchNorm1d(self.hidden_dims_predict[0]))
              
        for i in range(len(self.hidden_dims_predict) - 1):
            predictor.append(nn.Linear(self.hidden_dims_predict[i], self.hidden_dims_predict[i+1]))
            predictor.append(nn.ReLU())
            if self.batch_norm:
                predictor.append(nn.BatchNorm1d(self.hidden_dims_predict[i+1]))
             
        predictor.append(nn.Linear(self.hidden_dims_predict[-1], self.parameter_dim))
        
        self.predictor = nn.Sequential(*predictor)
 
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
                   
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, 'Adam')
        optimizer = optimizer(self.parameters(), self.learning_rate)
        
        scheduler = partial(getattr(torch.optim.lr_scheduler, self.scheduler), optimizer)
        scheduler = scheduler(**self.scheduler_params)

        return [optimizer], [scheduler]

    def _prepare_batch(self, batch):
        x, sig_x, y = batch
   
        return x, sig_x, y
        
    def _common_step(self, batch, batch_idx, stage: str):
        
        x, sig_x, y = self._prepare_batch(batch)
        
        if not self.train_predictor:
        
            x_recon = self(x)
            noise_floor = 1e-9

            loss = torch.sum(
                torch.log(sig_x**2 + noise_floor)/2
                + (x - x_recon)**2/(2*sig_x**2+noise_floor),
            )      

        else:
            with torch.no_grad():
                z = self.encoder(x)
                
            y_pred = self.predictor(z)
            loss = F.mse_loss(y_pred, y)
            
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss
