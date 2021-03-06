import numpy as np
import torch  

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pyrsd_emu import mlp
from pyrsd_emu import data_module
from pyrsd_emu import format_logger

import argparse
import logging

import os 
from pathlib import Path


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')

    parser.add_argument("--cosmo_to_pk", action="store_false",
                        help="Training SSL")

    parser.add_argument("--multipole_order", type=int, default=3,
                        help="Multipole orders to use for training [1,2,3]", choices=[1,2,3])

    parser.add_argument("--input_dim", type=int, default=7,
                        help="Input dimensionality")

    parser.add_argument("--hidden_dims", nargs='+', default=[256, 256, 256],
                        help="Input dimensionality")

    # Data loading
    parser.add_argument("--train_path", type=str, default='../data/pyrsd_training_set/powerspectra_7param_train.h5',
                        help="Path to hdf5 data file")

    parser.add_argument("--val_path", type=str, default='../data/pyrsd_training_set/powerspectra_7param_val.h5',
                        help="Path to hdf5 data file")
 
    parser.add_argument("--norm_path", type=str, default='../data/pyrsd_training_set/pk_nonlin_convolved_mean_std_normalization.txt',
                        help="Path to hdf5 data file")

    parser.add_argument("--covariance_path", type=str, default='../data/observed/covariance_matrix.npy',
                        help="Path to hdf5 data file")
                          
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of data loader workers")

    # Training 
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")

    parser.add_argument("--gpus", type=int, default=-1,
                        help="Number of gpus to use")

    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of gpu nodes available")

    # ddp does not work in ipython notebook, only ddp_spawn does
    parser.add_argument("--strategy", type=str, default='ddp', #'ddp_spawn',
                        help="training augmentations to use")

    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="How often to run validation epoch")
        
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=1,
                        help="Checkpoint model every n epochs")

    # Optimizers
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for model training")

    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for model optimization")

    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Max number of training epochs")

    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="Optimizer to use", choices=['Adam', 'SGD'])

    parser.add_argument("--T_max", type=int, default=100,
                        help="T_max for cosine learning rate scheduler")

    parser.add_argument("--seed", type=int , default=13579,
                        help="random seed for train test split")

    # Setup outputs and others
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Continue training from checkpoint on disk")

    parser.add_argument("--OUTPUT_DIR", type=str, default='../experiments/',
                        help="directory to save trained model and logs")

    parser.add_argument("--logfile_name", type=str, default='train_emulator.log',
                        help="name of log file")
        
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    return args

def main(args):
    params = vars(args)
    pl.seed_everything(params['seed'])

    params['cosmo_to_pk'] = True
    params['output_dim'] = 76*params['multipole_order'] 
    params['hidden_dims'] = [int(i) for i in params['hidden_dims']]
    
    params['OUTPUT_DIR'] = os.path.join(params['OUTPUT_DIR'], f"mlp_cosmo_to_pk_dims_{'_'.join(str(i) for i in params['hidden_dims'])}/")
    Path(params['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)

    # set up logging

    logger = format_logger.create_logger(
        filename=os.path.join(params['OUTPUT_DIR'], params['logfile_name']),
        )

    logger.info("\nTraining with the following parameters:")
    for k, v in params.items():
        logger.info(f"{k}: {v}")

    model = mlp.MLP(params)

    datamodule = data_module.PowerspectraDataModule(params)

    # # # Load covariance matrix
    # if not os.path.isfile(params['covariance_path']):
    #     covariance_matrix = torch.eye(params['output_dim']) 
    # else:
    #     covariance_matrix = torch.Tensor(np.load(params['covariance_path']).astype(np.float32))
    #     datamodule.normalize_pk(covariance_matrix, use_mean=False)

    #     covariance_matrix_inv = torch.linalg.inv(covariance_matrix)
    #     # self.covariance_matrix_inv = self.covariance_matrix_inv.cuda()

    # params['covariance_matrix'] = covariance_matrix
    # params['covariance_matrix_inv'] = covariance_matrix_inv


    if params['verbose']:
        print(model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=params['OUTPUT_DIR'],
        filename='{epoch:03d}-{val_loss:.6f}',
        every_n_epochs=params['checkpoint_every_n_epochs'],
        save_top_k=-1,
        save_on_train_epoch_end=True,
        verbose=True,
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(params['OUTPUT_DIR'], 'logs/'),
        name='mlp_cosmo_to_pk',
    )


    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
    )
    # trainer = pl.Trainer(
    #     max_epochs=params['max_epochs'],
    #     check_val_every_n_epoch=1,
    #     callbacks=[checkpoint_callback, lr_monitor],
    #     strategy=params['strategy'],
    #     logger=tb_logger,
    #     accelerator='cpu',
    #     gpus=-1,
    # )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=params['ckpt_path'],
    )

if __name__ == '__main__':

    args = parse_arguments()

    main(args)

