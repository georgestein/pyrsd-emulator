import numpy as np
import torch  
import h5py
import pytorch_lightning as pl

from typing import Any, Callable, Optional, List
import os

class PowerspectraDataset(torch.utils.data.Dataset):
    """
    Dataset class for powerspectra

    Data is a .npy array on disk

    Labels from external .npy array

    Parameters:
    -----------
    data_path:
        Path to powerspectra file
    norm_path:
        Path to file containing mean and std measured over training set
    transform:
        data augmentations to use
    """
    def __init__(
        self,
        data_path,
        norm_path,
        covariance_path,
        transform=None,
        output_dim=228,
        multipole_order=3,
        pk_type='nonlin_convolved'
    ):
        self.data_path  = data_path
        self.norm_path = norm_path
        self.covariance_path = covariance_path
        self.transforms = transform

        self.output_dim = output_dim
        self.multipole_order = multipole_order

        self.pk_type = pk_type

        # all 11 params
        # self.param_min = np.array([1.8, 1.2, 0.6, 1., 0., 0., 0.55, 1.15, 0.5, 0., 0.], dtype=np.float32)
        # self.param_max = np.array([3.0, 2.5, 1., 7., 0.25, 1., 2.35, 2.95, 0.9, 3., 18.], dtype=np.float32)

        # 7 params that actually have an effect
        self.param_min = np.array([1.2, 0.6, 0., 0., 0.5, 0., 0.], dtype=np.float32)
        self.param_max = np.array([2.5, 1., 0.25, 1., 0.9, 3., 18.], dtype=np.float32)
       
        # Load data normalizations
        if not os.path.isfile(self.norm_path):
            self.pk_mean, self.pk_std = 0., 1. 
        else:
            self.pk_mean, self.pk_std = np.loadtxt(self.norm_path, unpack=True)

        # Load covariance matrix
        if not os.path.isfile(self.covariance_path):
            self.covariance_matrix = torch.eye(params['output_dim']) 
        else:
            self.covariance_matrix = torch.Tensor(np.load(self.covariance_path).astype(np.float32))
            self.normalize_pk(self.covariance_matrix, use_mean=False)

            self.covariance_matrix_inv = torch.linalg.inv(self.covariance_matrix)

    
    def _open_file(self):
        self.hfile = h5py.File(self.data_path,'r')

    def __len__(self):
        with h5py.File(self.data_path, 'r') as hf:
            self.n_samples = hf[f'pk0_{self.pk_type}'].shape[0]
                
        return self.n_samples

    def normalize_model_params(self, params, inv=False):

        # if not inv:
        #     return (params - self.param_min)/(self.param_max - self.param_min) 

        # return params * (self.param_max - self.param_min) + self.param_min
 
        if not inv:
            return 2*(params - self.param_min)/(self.param_max - self.param_min) - 1

        return (params + 1)/2 * (self.param_max - self.param_min) + self.param_min
       
    def normalize_pk(self, pk, use_mean=True, inv=False):

        if not inv:
            if use_mean:
                return (pk - self.pk_mean)/self.pk_std
            else:
                return pk/self.pk_std

        if use_mean:
            return pk * self.pk_std + self.pk_mean
        else:
            return pk * self.pk_std

            
    def __getitem__(self, idx: int):

        if not hasattr(self, 'hfile'):
            self._open_file()

        # Concatenate desired powerspectra multipoles along last dimension
        y = np.hstack(
            [self.hfile[f"pk{i*2}_{self.pk_type}"][idx] for i in range(self.multipole_order)],
        )

        # If sig_y does not vary for each data sample
        # sig_y = np.hstack(
        #     [self.hfile[f"pk{i*2}_sig"] for i in range(self.multipole_order)],
        # )

        # if sig_y does vary
        # sig_y = np.hstack(
        #     [self.hfile[f"pk{i*2}_sig"][idx] for i in range(self.multipole_order)],
        # )
        
        # sig_y *= np.sqrt(1000)

        # y, sig_y = self.normalize_pk(y), self.normalize_pk(sig_y, use_mean=False)
        y = self.normalize_pk(y)

        x = self.hfile['model_params'][idx]
        x = self.normalize_model_params(x) 
        # x = self.transforms(x) 

        cov_inv = self.covariance_matrix_inv

        return x.astype(np.float32), y.astype(np.float32), cov_inv

class PowerspectraDataModule(pl.LightningDataModule):
    """
    Loads data from a single large hdf5 file,
    """
    
    def __init__(
        self,
        params: dict,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
    ) -> None:
        
        super().__init__()
        
        self.params = params
        
        self.train_path = self.params.get("train_path", "./")
        self.val_path = self.params.get("val_path", "./") # Set val to same as train
        self.norm_path = self.params.get("norm_path", "./") # path to file containing normalization measured over training set
        self.covariance_path = self.params.get("covariance_path", "./") # path to file containing normalization measured over training set

        self.multipole_order = self.params.get('multipole_order', 3)
        self.output_dim = self.params.get('output_dim', 228)

        self.num_workers = self.params.get("num_workers", 1)
        self.batch_size = self.params.get("batch_size", 4)
        
        self.shuffle = self.params.get("shuffle", True)
        self.pin_memory = self.params.get("pin_memory", True)
        self.drop_last = self.params.get("drop_last", True) # drop last due to queue_size % batch_size == 0. assert in Moco_v2
                
    def _default_transforms(self) -> Callable:

        # transform = DecalsTransforms(self.params)
        transform = None
        
        return transform    
    
    def prepare_data(self) -> None:

        if not os.path.isfile(self.train_path):
            raise FileNotFoundError(
                """
                Your training datafile cannot be found
                """
            )
             
    def setup(self, stage: Optional[str] = None):
        
        # Assign train/val datasets for use in dataloaders     
        if stage == "fit" or stage is None:
    
            train_transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms
            self.train_dataset = PowerspectraDataset(
                self.train_path,
                self.norm_path,
                self.covariance_path,
                train_transforms,
                output_dim=self.output_dim,
                multipole_order=self.multipole_order,
            )
            

            # Val set not used for now
            val_transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
            self.val_dataset = PowerspectraDataset(
                self.val_path,
                self.norm_path,
                self.covariance_path,
                val_transforms,
                output_dim=self.output_dim,
                multipole_order=self.multipole_order,
            )

        if stage == "predict" or stage is None:

            predict_transforms = CropTransform(self.params)
            # Predict over all training data
            self.predict_dataset = PowerspectraDataset(
                self.train_path,
                self.norm_path,
                self.covariance_path,
                predict_transforms,
                output_dim=self.output_dim,
                multipole_order=self.multipole_order,
            )

                    
    def train_dataloader(self):
 
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
            
        return loader
    
    def val_dataloader(self):

        loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )       
        
        return loader

    def predict_dataloader(self):
        
        loader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )       
        
        return loader
