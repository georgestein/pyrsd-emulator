from pytorch_lightning import loggers as pl_loggers

from functools import partial

multipole_order = 3
latent_dims = [4]#,3,4,5,6,7,8,9,10,11]

for latent_dim in latent_dims:
    
    pl.seed_everything(13579)

    params = {}
    # params['data_path'] = '../data/powerspectra.h5'
    params['train_path'] = '../data/powerspectra_11param_train.h5'
    params['val_path'] = '../data/powerspectra_11param_val.h5'

    params['max_epochs'] = 25
    params['batch_size'] = 32
    params['batch_norm'] = False

    params['num_workers'] = 4

    params['input_dim'] = 76*multipole_order #64
    params['hidden_dims'] = [128, 128, 128, 128]
    params['hidden_dims_predict'] = [8,8,8]
    params['latent_dim'] = latent_dim
    params['learning_rate'] = 0.0005
    params['T_max'] = 25

    params['optimizer'] = 'Adam'


    # Train AE Model
    output_path = f"../model_outputs/autoencoder_dims_{'_'.join(str(i) for i in params['hidden_dims'])}_{params['latent_dim']}/"

    model = AutoEncoder(params)

    datamodule = PowerspectraDataModule(params)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_path,
        filename='{epoch}-{val_loss:.2f}'
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(output_path, 'logs/'),
        name="log",
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
    
    # Train predictor (z -> model params) Model
    params['max_epochs'] = 5
    output_path = f"../model_outputs/autoencoder_predict_dims_{'_'.join(str(i) for i in params['hidden_dims'])}_{params['latent_dim']}/"
    model.train_predictor = True
    model.encoder.eval()
    
    datamodule = PowerspectraDataModule(params)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_path,
        filename='{epoch}-{val_loss:.2f}'
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(output_path, 'logs/'),
        name="log",
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
