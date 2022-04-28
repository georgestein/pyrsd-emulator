from pytorch_lightning import loggers as pl_loggers
pl.seed_everything(13579)

multipole_order = 3

params = {}
# params['data_path'] = '../data/powerspectra.h5'
params['train_path'] = '../data/powerspectra_11param_train.h5'
params['val_path'] = '../data/powerspectra_11param_val.h5'
params['max_epochs'] = 25
params['batch_size'] = 32
params['num_workers'] = 4

params['input_dim'] = 76*multipole_order
params['hidden_dims'] = [128, 128, 128, 128]

params['T_max'] = 100
params['learning_rate'] = 0.0005
params['optimizer'] = 'Adam'

params['output_dim'] = 11
params['cosmo_to_pk'] = False

model = MLP(params)

datamodule = PowerspectraDataModule(params)

lr_monitor = pl.callbacks.LearningRateMonitor(
    logging_interval='epoch',
)

tb_logger = pl_loggers.TensorBoardLogger(
    save_dir='logs',
    name='mlp_pk_to_cosmo',
)

trainer = pl.Trainer(
    max_epochs=params['max_epochs'],
    check_val_every_n_epoch=1,
    callbacks=[lr_monitor],
    # strategy="ddp_spawn",
    logger=tb_logger,
    gpus=1,
)

trainer.fit(
    model,
    datamodule=datamodule,
)
