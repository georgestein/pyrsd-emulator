#!/usr/bin/env python
# Import
from pyRSD import pygcl
import matplotlib.pyplot as plt

# from nbodykit.lab import *
import numpy as np
from pyRSD.rsdfit.data import PowerMeasurements
from pyRSD.rsdfit import FittingDriver
from pyRSD.rsd.transfers import WindowFunctionTransfer

from multiprocessing import Pool

import h5py
import time
import os
import math
import sys

nsamples = 256000 #256000 #128000
chunksize = 100
print_every = 100
pool_size = 64

ells = [0, 2, 4]
Nk = 512
Nmu = 50

kmin=0.001
kmax=1.0
   
mu = np.linspace(0, 1., Nmu)
k_pole_pkmu = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

nowiggle = False
# nowiggle = True

# Result file path
results_file = '2022-02-22_nlopt_59i_chain0__1.npz'
model_file = 'Run1_Model_LOWZ_N.npy'
dirname='/pscratch/sd/g/gstein/machine_learning/pae/pyrsd_emulator/data/construct_pyrsd_data'

Q = np.loadtxt('wcp_rd10_DDsmu_PATCHY_6C_wxw_North_0.2-0.5_normed.dat')

nchunks = math.ceil(nsamples/chunksize)

def get_Pells(d, k, multipole_orders=[0, 2, 4]):

    Pell = np.stack([d.theory.model.poles(k, m)[:,0] for m in multipole_orders])
    
    return Pell

def get_Pells_lin(d, k):
    cosmo_d = d.theory.model.cosmo
    if nowiggle:
        cosmo_d = cosmo_d.clone(tf=pygcl.transfers.EH_NoWiggle)
    linpower_z1 = pygcl.LinearPS(cosmo_d, d.model.z)
    linpower_z1.SetSigma8AtZ(d.theory.fit_params['sigma8_z'].value)

    b = d.theory.fit_params['b1'].value
    f = d.theory.fit_params['f'].value

    lin = linpower_z1(k)
    lin0 = (b**2 + 2*b*f/3 + (1/5)*f**2)*lin
    lin2 = (4*b*f/3 + (4/7)*f**2)*lin  
    lin4 = (8/35*f**2)*lin  

    Pell_lin = np.stack([lin0, lin2, lin4])
    
    return Pell_lin

def convolve_Pell(d, k):
    
    # P(k,mu)
    Pkmu = d.theory.model.power(k_pole_pkmu, mu)

    # P_ell's
    Pell_convolved = transfer(Pkmu, k).T
    
    return Pell_convolved

# Pell = get_Pells(d, k)
# Pell_convolved = convolve_Pell(d, k)
# Pell_lin = get_Pells_lin(d, k)

# print(Pell.shape, Pell_convolved.shape, Pell_lin.shape)


# Sample new model parameters from prior
def sample_params_from_priors(params: dict):
    """
    params is a dictionary with parameter name as key, and prior/prior_vals as items
    """
    sample = {}
    for k in params:
        if params[k]['prior'] == 'uniform':
            lower, upper = params[k]['prior_vals']
            val = np.random.uniform(lower, upper)
        if params[k]['prior'] == 'normal':
            mu, sig = params[k]['prior_vals']
            val = np.random.normal(mu, sig)
                       
        sample[k] = val
        
    return sample

def update_model_params(sample, model):
    for param in sample:

        if param =='Nsat_mult': 
            model.Nsat_mult = sample[param]

        if param == 'b1_cA': 
            model.b1_cA = sample[param]

        if param == 'f': 
            model.f = sample[param]

        if param == 'f1h_sBsB':
            model.f1h_sBsB = sample[param]

        if param == 'fs':
            model.fs = sample[param]

        if param == 'fsB': 
            model.fsB = sample[param]

        if param == 'gamma_b1sA': 
            model.gamma_b1sA = sample[param]

        if param == 'gamma_b1sB': 
            model.gamma_b1sB = sample[param]

        if param == 'sigma8_z': 
            model.sigma8_z = sample[param]

        if param == 'sigma_c': 
            model.sigma_c = sample[param]

        if param == 'sigma_sA': 
            model.sigma_sA = sample[param]

    # return model

def update_fitted_model_params(sample, model):
    for param in sample:
        model[param].value = sample[param]


def generate_chunk(ichunk):
    
    data_dir = '../powerspectra_chunks/'
    # data_path = f'{data_dir}/powerspectra_{param_string}.h5'
    data_path = f'{data_dir}/powerspectra_11param_{ichunk*chunksize:07d}_{(ichunk+1)*chunksize:07d}.h5'

    if os.path.isfile(data_path):
        print(f"{ichunk} already exists, skipping")
        return

    print(f"running on chunk {ichunk}")
    np.random.seed(ichunk)

    # Specify model + result file paths
    d = FittingDriver.from_directory(dirname=dirname, model_file=model_file, results_file=results_file)
    d.set_fit_results()

    # Get params and their priors and put in dictionary
    params = d.theory.fit_params
    # use all varied params
    names  = [k for k in params if params[k].vary] 

    # names = ['sigma8_z', 'f', 'b1_cA']
    values = [params[i].value for i in names] 
    priors = [params[i].prior for i in names] 

    param_priors = {}
    for i in range(len(names)):
        
        name = names[i]
        value = values[i]
        prior = priors[i]

        try:
            prior_type = 'normal'
            prior_vals = [prior.mu, prior.sigma]
        except:
            prior_type = 'uniform'
            prior_vals = [prior.lower, prior.upper]
            
        param_priors[name] = {
            'value': value,
            'prior': prior_type,
            'prior_vals': prior_vals,
        }
        
    param_priors['sigma8_z']['prior_vals'][0] = 0.5

    param_string = '-'.join(list(param_priors.keys())) 

    # Set no wiggle transfer function, as we will be adding wiggles back in during inference (they only minimally change results)
    # Set the transfer function as Eisenstein & Hu + no wiggle
    if nowiggle:
        d.theory.model.cosmo.SetTransferFunction(tf=pygcl.transfers.EH_NoWiggle)

    # P(k) best-fit model
    model_d = d.theory.model
    cosmo = d.theory.model.cosmo
    cosmo_d = d.theory.model.cosmo
   
    linpower_z1 = pygcl.LinearPS(cosmo_d, d.model.z)
    linpower_z1.SetSigma8AtZ(d.theory.fit_params['sigma8_z'].value)

    # Get k-bins
    kbin_data = np.load("../p0-p2-p4_cov.npz")
    k         = kbin_data['k'].astype(np.float32)
    P_err     = kbin_data['sigma'].astype(np.float32)

    nk   = k.shape[0]

    # For colvolvd powerspectra

    # Use a transfer function to convolve with the window function
    transfer = WindowFunctionTransfer(Q, ells, kmin=kmin, kmax=kmax, Nk=Nk, Nmu=Nmu, max_ellprime=4)
    # transfer = WindowFunctionTransfer(Q, ells, kmin=1e-4, kmax=1, Nk = Nk, Nmu = Nmu, max_ellprime = 4)

    model_params = np.empty((chunksize, len(param_priors)), dtype=np.float32)

    pk0_lin    = np.empty((chunksize, nk), dtype=np.float32)
    pk2_lin    = np.empty((chunksize, nk), dtype=np.float32)
    pk4_lin    = np.empty((chunksize, nk), dtype=np.float32)
    pk0_nonlin = np.empty((chunksize, nk), dtype=np.float32)
    pk2_nonlin = np.empty((chunksize, nk), dtype=np.float32)
    pk4_nonlin = np.empty((chunksize, nk), dtype=np.float32)
    pk0_nonlin_convolved = np.empty((chunksize, nk), dtype=np.float32)
    pk2_nonlin_convolved = np.empty((chunksize, nk), dtype=np.float32)
    pk4_nonlin_convolved = np.empty((chunksize, nk), dtype=np.float32)

    # pk0_r = np.empty((nsamples, nk), dtype=np.float32)
    # pk2_r = np.empty((nsamples, nk), dtype=np.float32)

    tstart = time.time()
    for i in range(chunksize):
        # print(ichunk, i)
        sample = sample_params_from_priors(param_priors)
        model_params[i] = list(sample.values())

        update_fitted_model_params(sample, d.theory.fit_params)
        update_model_params(sample, model_d)

        model_d.b1 = (1 - model_d.fs)*model_d.b1_c + model_d.fs*model_d.b1_s
        d.theory.fit_params['b1'].value = (1 - model_d.fs)*model_d.b1_c + model_d.fs*model_d.b1_s

        b = model_d.b1
        f = model_d.f    

        linpower_z1 = pygcl.LinearPS(cosmo_d, d.model.z)
        linpower_z1.SetSigma8AtZ(model_d.sigma8_z)

        # Re-load best-fit (unconvolved) P_ell's
        Pell = get_Pells(d, k)
        # print('got pell')
        # P(k,mu)
        Pkmu = d.theory.model.power(k_pole_pkmu, mu)

        # P_ell's
        Pell_convolved = transfer(Pkmu, k).T
        # Pell_convolved = convolve_Pell(d, k)
        # print('got pell convolved')

        lin = linpower_z1(k)
        lin0 = (b**2 + 2*b*f/3 + (1/5)*f**2)*lin
        lin2 = (4*b*f/3 + (4/7)*f**2)*lin  
        lin4 = (8/35*f**2)*lin  

        Pell_lin = np.stack([lin0, lin2, lin4])
        # Pell_ratio = Pell/Pell_lin    
        # Pell_ratio_sig = np.abs(Pell_ratio) * np.sqrt( (P_err/Pell)**2 + (P_err/Pell_lin)**2)
        # print('got pell lin')

        # Non-linear spectra
        pk0_lin[i], pk2_lin[i], pk4_lin[i] = Pell_lin[0], Pell_lin[1], Pell_lin[2]
        pk0_nonlin[i], pk2_nonlin[i], pk4_nonlin[i] = Pell[0], Pell[1], Pell[2]
        pk0_nonlin_convolved[i], pk2_nonlin_convolved[i], pk4_nonlin_convolved[i] = Pell_convolved[0], Pell_convolved[1], Pell_convolved[2]

        # if i % print_every == 0:
        #     print(f"{ichunk}, {i}: time elapsed = {time.time() - tstart}")

    pk0_r = pk0_nonlin/pk0_lin
    pk2_r = pk2_nonlin/pk2_lin
    pk4_r = pk4_nonlin/pk4_lin

    pk0_r_sig = np.abs(pk0_r) * np.sqrt( (P_err[0]/pk0_lin)**2 + (P_err[0]/pk0_nonlin)**2)
    pk2_r_sig = np.abs(pk2_r) * np.sqrt( (P_err[1]/pk2_lin)**2 + (P_err[1]/pk2_nonlin)**2)
    pk4_r_sig = np.abs(pk4_r) * np.sqrt( (P_err[2]/pk4_lin)**2 + (P_err[2]/pk4_nonlin)**2)
    
    tend = time.time()
    print(f"SAVING DATA {ichunk}. Time elapsed: {tend-tstart}")

    # try: 
    #     os.remove(data_path)
    # except:
    #     print('file doesnt exist')

    with h5py.File(data_path, 'w') as f:
        f.create_dataset("k", data=k)    
        f.create_dataset("model_params", data=model_params)
        f.create_dataset("model_param_names", data=np.array(list(param_priors.keys())).astype('S'))
        
        f.create_dataset("pk0_sig", data=P_err[0])
        f.create_dataset("pk2_sig", data=P_err[1])
        f.create_dataset("pk4_sig", data=P_err[2])

        f.create_dataset("pk0_lin", data=pk0_lin)
        f.create_dataset("pk2_lin", data=pk2_lin)
        f.create_dataset("pk4_lin", data=pk4_lin)
        
        f.create_dataset("pk0_nonlin", data=pk0_nonlin)
        f.create_dataset("pk2_nonlin", data=pk2_nonlin)
        f.create_dataset("pk4_nonlin", data=pk4_nonlin)

        f.create_dataset("pk0_nonlin_convolved", data=pk0_nonlin_convolved)
        f.create_dataset("pk2_nonlin_convolved", data=pk2_nonlin_convolved)
        f.create_dataset("pk4_nonlin_convolved", data=pk4_nonlin_convolved)

        f.create_dataset("pk0_r", data=pk0_r)
        f.create_dataset("pk2_r", data=pk2_r)
        f.create_dataset("pk4_r", data=pk4_r)

        f.create_dataset("pk0_r_sig", data=pk0_r_sig)
        f.create_dataset("pk2_r_sig", data=pk2_r_sig)
        f.create_dataset("pk4_r_sig", data=pk4_r_sig)

    return

# for i in range(nchunks):
#     generate_chunk(i)
pool = Pool(processes=pool_size)
with pool:
    pool.map(generate_chunk, range(nchunks))

# def main():
#     pool = Pool(processes=pool_size)
#     with pool:
#         pool.imap(generate_chunk, range(nchunks))
    
# if __name__ == '__main__':
#     main()

# pool = Pool(pool_size)
# with pool:
#     pool.map(generate_chunk, range(nchunks))
    
# pool.close()
# pool.join()

