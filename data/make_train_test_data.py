"""
Compile all chunks of data into train/test/val h5py datafiles
"""
import numpy as np
import h5py
import glob


files = sorted(glob.glob("powerspectra_chunks/powepowerspectra_11param_0*"))

print(files)
# with h5py.File(files[0], 'r') as hf:    
#     chunksize = hf['model_params'].shape[0]
#     nparams = hf['model_params'].shape[1]
#     nkbins = hf['pk0_lin'].shape[1]
#     k = hf['k'][:]
#     model_param_names = hf['model_param_names'][:]
    
# nsamples = chunksize*len(files)
# print(f"nsamples={nsamples}")
# model_params = np.empty((nsamples, nparams), dtype=np.float32)
# pk0_sig = np.empty((nsamples, nkbins), dtype=np.float32)
# pk2_sig = np.empty((nsamples, nkbins), dtype=np.float32)
# pk4_sig = np.empty((nsamples, nkbins), dtype=np.float32)
# pk0_lin = np.empty((nsamples, nkbins), dtype=np.float32)
# pk2_lin = np.empty((nsamples, nkbins), dtype=np.float32)
# pk4_lin = np.empty((nsamples, nkbins), dtype=np.float32)
# pk0_nonlin = np.empty((nsamples, nkbins), dtype=np.float32)
# pk2_nonlin = np.empty((nsamples, nkbins), dtype=np.float32)
# pk4_nonlin = np.empty((nsamples, nkbins), dtype=np.float32)
# pk0_r = np.empty((nsamples, nkbins), dtype=np.float32)
# pk2_r = np.empty((nsamples, nkbins), dtype=np.float32)
# pk4_r = np.empty((nsamples, nkbins), dtype=np.float32)
# pk0_r_sig = np.empty((nsamples, nkbins), dtype=np.float32)
# pk2_r_sig = np.empty((nsamples, nkbins), dtype=np.float32)
# pk4_r_sig = np.empty((nsamples, nkbins), dtype=np.float32)

# for i, f in enumerate(files):
#     with h5py.File(f, 'r') as hf:    
#         istart = i*chunksize
#         iend = (i+1)*chunksize
        
#         model_params[istart:iend] = hf['model_params'][:]
#         pk0_sig[istart:iend] = hf['pk0_sig'][:]
#         pk2_sig[istart:iend] = hf['pk2_sig'][:]
#         pk4_sig[istart:iend] = hf['pk4_sig'][:]
#         pk0_lin[istart:iend] = hf['pk0_lin'][:]
#         pk2_lin[istart:iend] = hf['pk2_lin'][:]
#         pk4_lin[istart:iend] = hf['pk4_lin'][:]
#         pk0_nonlin[istart:iend] = hf['pk0_nonlin'][:]
#         pk2_nonlin[istart:iend] = hf['pk2_nonlin'][:]
#         pk4_nonlin[istart:iend] = hf['pk4_nonlin'][:]
#         pk0_r[istart:iend] = hf['pk0_r'][:]
#         pk2_r[istart:iend] = hf['pk2_r'][:]
#         pk4_r[istart:iend] = hf['pk4_r'][:]
#         pk0_r_sig[istart:iend] = hf['pk0_r_sig'][:]
#         pk2_r_sig[istart:iend] = hf['pk2_r_sig'][:]
#         pk4_r_sig[istart:iend] = hf['pk4_r_sig'][:]
                

# data_path_out = '../data/powerspectra_11param_all.h5'
# with h5py.File(data_path_out, 'w') as hf:   
    
#     hf.create_dataset("model_param_names", data=model_param_names)   
#     hf.create_dataset("k", data=k)    
#     hf.create_dataset("model_params", data=model_params)  
#     hf.create_dataset("pk0_sig", data=pk0_sig)
#     hf.create_dataset("pk2_sig", data=pk2_sig)
#     hf.create_dataset("pk4_sig", data=pk4_sig)
#     hf.create_dataset("pk0_lin", data=pk0_lin)
#     hf.create_dataset("pk2_lin", data=pk2_lin)
#     hf.create_dataset("pk4_lin", data=pk4_lin)
#     hf.create_dataset("pk0_nonlin", data=pk0_nonlin)
#     hf.create_dataset("pk2_nonlin", data=pk2_nonlin)
#     hf.create_dataset("pk4_nonlin", data=pk4_nonlin)
#     hf.create_dataset("pk0_r", data=pk0_r)
#     hf.create_dataset("pk2_r", data=pk2_r)
#     hf.create_dataset("pk4_r", data=pk4_r)
#     hf.create_dataset("pk0_r_sig", data=pk0_r_sig)
#     hf.create_dataset("pk2_r_sig", data=pk2_r_sig)
#     hf.create_dataset("pk4_r_sig", data=pk4_r_sig)
