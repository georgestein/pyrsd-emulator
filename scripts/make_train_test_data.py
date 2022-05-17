import numpy as np
import h5py
import glob

files = sorted(glob.glob("../data/powerspectra_chunks/powerspectra_11param_*"))

with h5py.File(files[0], 'r') as hf:    
    chunksize = hf['model_params'].shape[0]
    nparams = hf['model_params'].shape[1]
    nkbins = hf['pk0_lin'].shape[1]
    k = hf['k'][:]
    model_param_names = hf['model_param_names'][:]
    
nsamples = chunksize*len(files)
print(f"Nfiles: {len(files)}, N samples per file={chunksize}, nsamples={nsamples}")

model_params = np.empty((nsamples, nparams), dtype=np.float32)
pk0_sig = np.empty((nsamples, nkbins), dtype=np.float32)
pk2_sig = np.empty((nsamples, nkbins), dtype=np.float32)
pk4_sig = np.empty((nsamples, nkbins), dtype=np.float32)
pk0_lin = np.empty((nsamples, nkbins), dtype=np.float32)
pk2_lin = np.empty((nsamples, nkbins), dtype=np.float32)
pk4_lin = np.empty((nsamples, nkbins), dtype=np.float32)
pk0_nonlin = np.empty((nsamples, nkbins), dtype=np.float32)
pk2_nonlin = np.empty((nsamples, nkbins), dtype=np.float32)
pk4_nonlin = np.empty((nsamples, nkbins), dtype=np.float32)
pk0_nonlin_convolved = np.empty((nsamples, nkbins), dtype=np.float32)
pk2_nonlin_convolved = np.empty((nsamples, nkbins), dtype=np.float32)
pk4_nonlin_convolved = np.empty((nsamples, nkbins), dtype=np.float32)
pk0_r = np.empty((nsamples, nkbins), dtype=np.float32)
pk2_r = np.empty((nsamples, nkbins), dtype=np.float32)
pk4_r = np.empty((nsamples, nkbins), dtype=np.float32)
pk0_r_sig = np.empty((nsamples, nkbins), dtype=np.float32)
pk2_r_sig = np.empty((nsamples, nkbins), dtype=np.float32)
pk4_r_sig = np.empty((nsamples, nkbins), dtype=np.float32)

for i, f in enumerate(files):
    with h5py.File(f, 'r') as hf:    
        istart = i*chunksize
        iend = (i+1)*chunksize
        
        model_params[istart:iend] = hf['model_params'][:]
        pk0_sig[istart:iend] = hf['pk0_sig'][:]
        pk2_sig[istart:iend] = hf['pk2_sig'][:]
        pk4_sig[istart:iend] = hf['pk4_sig'][:]
        pk0_lin[istart:iend] = hf['pk0_lin'][:]
        pk2_lin[istart:iend] = hf['pk2_lin'][:]
        pk4_lin[istart:iend] = hf['pk4_lin'][:]
        pk0_nonlin[istart:iend] = hf['pk0_nonlin'][:]
        pk2_nonlin[istart:iend] = hf['pk2_nonlin'][:]
        pk4_nonlin[istart:iend] = hf['pk4_nonlin'][:]
        pk0_nonlin_convolved[istart:iend] = hf['pk0_nonlin_convolved'][:]
        pk2_nonlin_convolved[istart:iend] = hf['pk2_nonlin_convolved'][:]
        pk4_nonlin_convolved[istart:iend] = hf['pk4_nonlin_convolved'][:]
        pk0_r[istart:iend] = hf['pk0_r'][:]
        pk2_r[istart:iend] = hf['pk2_r'][:]
        pk4_r[istart:iend] = hf['pk4_r'][:]
        pk0_r_sig[istart:iend] = hf['pk0_r_sig'][:]
        pk2_r_sig[istart:iend] = hf['pk2_r_sig'][:]
        pk4_r_sig[istart:iend] = hf['pk4_r_sig'][:]
                

data_path_out = '../data/powerspectra_11param_all.h5'
with h5py.File(data_path_out, 'w') as hf:   
    
    hf.create_dataset("model_param_names", data=model_param_names)   
    hf.create_dataset("k", data=k)    
    hf.create_dataset("model_params", data=model_params)  
    hf.create_dataset("pk0_sig", data=pk0_sig)
    hf.create_dataset("pk2_sig", data=pk2_sig)
    hf.create_dataset("pk4_sig", data=pk4_sig)
    hf.create_dataset("pk0_lin", data=pk0_lin)
    hf.create_dataset("pk2_lin", data=pk2_lin)
    hf.create_dataset("pk4_lin", data=pk4_lin)
    hf.create_dataset("pk0_nonlin", data=pk0_nonlin)
    hf.create_dataset("pk2_nonlin", data=pk2_nonlin)
    hf.create_dataset("pk4_nonlin", data=pk4_nonlin)
    hf.create_dataset("pk0_nonlin_convolved", data=pk0_nonlin_convolved)
    hf.create_dataset("pk2_nonlin_convolved", data=pk2_nonlin_convolved)
    hf.create_dataset("pk4_nonlin_convolved", data=pk4_nonlin_convolved)
    hf.create_dataset("pk0_r", data=pk0_r)
    hf.create_dataset("pk2_r", data=pk2_r)
    hf.create_dataset("pk4_r", data=pk4_r)
    hf.create_dataset("pk0_r_sig", data=pk0_r_sig)
    hf.create_dataset("pk2_r_sig", data=pk2_r_sig)
    hf.create_dataset("pk4_r_sig", data=pk4_r_sig)

data_path_in = '../data/powerspectra_11param_all.h5'

val_frac = 0.2

with h5py.File(data_path_in, 'r') as hf:
    ntotal = hf['pk0_r'].shape[0]
    
ntrain = int(ntotal * (1-val_frac))
nval = ntotal - ntrain
# DEBUG
# ntrain = 25000
# nval = 5000
# Write training
data_path_out = '../data/powerspectra_11param_train.h5'
with h5py.File(data_path_in, 'r') as hf:    
    with h5py.File(data_path_out, 'w') as f:
        f.create_dataset("model_param_names", data=hf['model_param_names'])   
        f.create_dataset("k", data=hf['k'][:])    
        f.create_dataset("model_params", data=hf['model_params'][:ntrain])   
        f.create_dataset("pk0_sig", data=hf['pk0_sig'][:ntrain])
        f.create_dataset("pk2_sig", data=hf['pk2_sig'][:ntrain])
        f.create_dataset("pk4_sig", data=hf['pk4_sig'][:ntrain])
        f.create_dataset("pk0_lin", data=hf['pk0_lin'][:ntrain])
        f.create_dataset("pk2_lin", data=hf['pk2_lin'][:ntrain])
        f.create_dataset("pk4_lin", data=hf['pk4_lin'][:ntrain])
        f.create_dataset("pk0_nonlin", data=hf['pk0_nonlin'][:ntrain])
        f.create_dataset("pk2_nonlin", data=hf['pk2_nonlin'][:ntrain])
        f.create_dataset("pk4_nonlin", data=hf['pk4_nonlin'][:ntrain])
        f.create_dataset("pk0_nonlin_convolved", data=hf['pk0_nonlin_convolved'][:ntrain])
        f.create_dataset("pk2_nonlin_convolved", data=hf['pk2_nonlin_convolved'][:ntrain])
        f.create_dataset("pk4_nonlin_convolved", data=hf['pk4_nonlin_convolved'][:ntrain])
        f.create_dataset("pk0_r", data=hf['pk0_r'][:ntrain])
        f.create_dataset("pk2_r", data=hf['pk2_r'][:ntrain])
        f.create_dataset("pk4_r", data=hf['pk4_r'][:ntrain])
        f.create_dataset("pk0_r_sig", data=hf['pk0_r_sig'][:ntrain])
        f.create_dataset("pk2_r_sig", data=hf['pk2_r_sig'][:ntrain])
        f.create_dataset("pk4_r_sig", data=hf['pk4_r_sig'][:ntrain])


# Write val
data_path_out = '../data/powerspectra_11param_val.h5'
with h5py.File(data_path_in, 'r') as hf:    
    with h5py.File(data_path_out, 'w') as f:
        f.create_dataset("model_param_names", data=hf['model_param_names'])   
        f.create_dataset("k", data=hf['k'][:])    
        f.create_dataset("model_params", data=hf['model_params'][ntrain:ntrain+nval])   
        f.create_dataset("pk0_sig", data=hf['pk0_sig'][ntrain:ntrain+nval])
        f.create_dataset("pk2_sig", data=hf['pk2_sig'][ntrain:ntrain+nval])
        f.create_dataset("pk4_sig", data=hf['pk4_sig'][ntrain:ntrain+nval])
        f.create_dataset("pk0_lin", data=hf['pk0_lin'][ntrain:ntrain+nval])
        f.create_dataset("pk2_lin", data=hf['pk2_lin'][ntrain:ntrain+nval])
        f.create_dataset("pk4_lin", data=hf['pk4_lin'][ntrain:ntrain+nval])
        f.create_dataset("pk0_nonlin", data=hf['pk0_nonlin'][ntrain:ntrain+nval])
        f.create_dataset("pk2_nonlin", data=hf['pk2_nonlin'][ntrain:ntrain+nval])
        f.create_dataset("pk4_nonlin", data=hf['pk4_nonlin'][ntrain:ntrain+nval])
        f.create_dataset("pk0_nonlin_convolved", data=hf['pk0_nonlin_convolved'][ntrain:ntrain+nval])
        f.create_dataset("pk2_nonlin_convolved", data=hf['pk2_nonlin_convolved'][ntrain:ntrain+nval])
        f.create_dataset("pk4_nonlin_convolved", data=hf['pk4_nonlin_convolved'][ntrain:ntrain+nval])
        f.create_dataset("pk0_r", data=hf['pk0_r'][ntrain:ntrain+nval])
        f.create_dataset("pk2_r", data=hf['pk2_r'][ntrain:ntrain+nval])
        f.create_dataset("pk4_r", data=hf['pk4_r'][ntrain:ntrain+nval])
        f.create_dataset("pk0_r_sig", data=hf['pk0_r_sig'][ntrain:ntrain+nval])
        f.create_dataset("pk2_r_sig", data=hf['pk2_r_sig'][ntrain:ntrain+nval])
        f.create_dataset("pk4_r_sig", data=hf['pk4_r_sig'][ntrain:ntrain+nval])

