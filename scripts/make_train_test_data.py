"""
Compile chunked data into final train/test/val sets
"""
import numpy as np
import h5py
import glob

val_frac = 0.2
test_frac = 0.1

files = sorted(glob.glob("../data/powerspectra_chunks/powerspectra_11param_*"))
data = {}
data_dims = {}
with h5py.File(files[0], 'r') as hf:    

    keys = list(hf.keys())
    chunksize = hf['model_params'].shape[0]

    for k in keys:
        s = hf[k].shape[:]

        if len(s) == 1: # data is the same for all files
            data[k] = hf[k][:]

        else: 
            data_dims[k] = s

   
nsamples = chunksize*len(files)
print(f"Nfiles: {len(files)}, N samples per file={chunksize}, nsamples={nsamples}")


for k in data_dims:
    data[k] = np.empty([nsamples] + [i for i in data_dims[k][1:]], dtype=np.float32)

# Loop through all files and load into arrays
for i, f in enumerate(files):
    if i % 100 == 0:
        print(f"loading from {i}th file")
    with h5py.File(f, 'r') as hf:    
        istart = i*chunksize
        iend = (i+1)*chunksize
        
        for k in data_dims:
            data[k][istart:iend] = hf[k][:]


# Save data in large h5 files
def save_dataset(ind_start, ind_end, filepath):
    with h5py.File(filepath, 'w') as hf:   
        for k in data:
            if data[k].shape[0] == nsamples:
                hf.create_dataset(k, data=data[k][ind_start:ind_end])  
            else:
                hf.create_dataset(k, data=data[k])   
 

ntest = int(nsamples*test_frac)
nval = int(nsamples*val_frac)
ntrain = nsamples - ntest - nval
print("ntrain, nval, ntest, nsamples = ", ntrain, nval, ntest, nsamples)
data_path_out = '../data/powerspectra_7param_all.h5'
save_dataset(0, nsamples, data_path_out)

# Data is already randomly shuffled - no need to do it again
data_path_out = '../data/powerspectra_7param_train.h5'
save_dataset(0, ntrain, data_path_out)

data_path_out = '../data/powerspectra_7param_val.h5'
save_dataset(ntrain, ntrain+nval, data_path_out)

data_path_out = '../data/powerspectra_7param_test.h5'
# save_dataset(ntrain+nval, nsamples, data_path_out)
save_dataset(nsamples-100, nsamples, data_path_out)

