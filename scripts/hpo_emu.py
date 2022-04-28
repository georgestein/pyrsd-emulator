# Optuna Hyparameter Sweep

import optuna
from optuna.trial import TrialState
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning import loggers as pl_loggers
pl.seed_everything(13579)

# params['data_path'] = '../data/powerspectra.h5'
      
def objective(trial):
    params = {}
    params['train_path'] = '../data/powerspectra_11param_train.h5'
    params['val_path'] = '../data/powerspectra_11param_val.h5'
    
    params['max_epochs'] = 10
    params['num_workers'] = 4
    params['output_dim'] = 76*3 #64
    params['cosmo_to_pk'] = True
    params['input_dim'] = 11
    
    params['batch_size'] = 128 #trial.suggest_int('batch_size', 16, 128)

    n_layers = trial.suggest_int("n_layers", 2, 3)
    params['hidden_dims'] = [trial.suggest_int(f"n_units_{i}", 8, 64) for i in range(n_layers)]
    params['learning_rate'] = trial.suggest_float("lr", 1e-3, 1e-1, log=True)

    params['optimizer'] = trial.suggest_categorical("optimizer", ["Adam"])#, "SGD"])
    # self.scheduler = self.hparams.get('scheduler', 'CosineAnnealingLR')
    # params['input_dim'] = 3
    # params['output_dim'] = 100
    # params['cosmo_to_pk'] = True


    model = MLP(params)
    print(params)
    print(model)
    
    datamodule = PowerspectraDataModule(params)

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    cb_pruning = PyTorchLightningPruningCallback(trial, monitor="loss")
    
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs',
        name='mlp_cosmo_to_pk',
    )
    # logger = DictLogger(trial.number)

    trainer = pl.Trainer(
        max_epochs=params['max_epochs'],
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor, cb_pruning],
        # strategy="ddp_spawn",
        logger=tb_logger,
        gpus=1,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )
    
    score = trainer.validate(
        model,
        datamodule=datamodule,
    )

    return score[0]['val_loss']

SEED = 13579
study = optuna.create_study(
    study_name='test',
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study.optimize(objective, n_trials=10, timeout=600)
