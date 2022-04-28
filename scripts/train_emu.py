import numpy as np
import torch  

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pyrsd_emu import mlp
from pyrsd_emu import data_module

import os 

pl.seed_everything(13579)


params = {}
# params['data_path'] = '../data/powerspectra.h5'
params['train_path'] = '../data/powerspectra_11param_train.h5'
params['val_path'] = '../data/powerspectra_11param_val.h5'

params['cosmo_to_pk'] = True
params['output_dim'] = 76*params['multipole_order'] 

params['max_epochs'] = 1
params['batch_size'] = 32
# params['batch_norm'] = True
params['num_workers'] = 4

params['multipole_order'] = 3
params['input_dim'] = 11
params['hidden_dims'] = [128, 128, 128, 128]

params['T_max'] = 100
params['learning_rate'] = 0.0005
params['optimizer'] = 'Adam'

output_path = f"../model_outputs/mlp_cosmo_to_pk_dims_{'_'.join(str(i) for i in params['hidden_dims'])}/"

model = mlp.MLP(params)

datamodule = data_module.PowerspectraDataModule(params)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=output_path,
    filename='{epoch}-{val_loss:.2f}'
)

lr_monitor = pl.callbacks.LearningRateMonitor(
    logging_interval='epoch',
)

tb_logger = pl_loggers.TensorBoardLogger(
    save_dir=os.path.join(output_path, 'logs/'),
    name='mlp_cosmo_to_pk',
)

trainer = pl.Trainer(
    max_epochs=params['max_epochs'],
    check_val_every_n_epoch=1,
    callbacks=[checkpoint_callback, lr_monitor],
    # strategy="ddp_spawn",
    logger=tb_logger,
    gpus=1,
)

trainer.fit(
    model,
    datamodule=datamodule,
)

# p4: loss=-8.07e+03
# All: loss=5e5
