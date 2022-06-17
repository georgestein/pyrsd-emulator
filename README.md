# pyrsd_emulator
emulator for pyrsd, with two main functionalities

1. Train a MLP to predict powerspectra multipoles (228 dim) from input params (7 dim) (see `scripts/train_emu.py`). Using PyTorch-Lightning for model training.
2. Construct a JAX model by loading in pytorch weights and biases, then perform NUTS sampling with NumPyro (see `notebooks/posterior_analysis.ipynb`). With a 4 layer MLP model, 256 neurons per layer, 10,000 samples takes about 2 minutes on CPU (slower on GPU - 30 mins!).


# Installation 

Codes are running on Perlmutter. JAX can be tricky to get working, espically on GPU. I was only able to get it working following these instructions (see <https://docs.nersc.gov/development/languages/python/using-python-perlmutter/> for more detail on this step):

	module load cudatoolkit/11.5
	module load cudnn/8.3.2
	module load python
	
	# Create a new conda environment 
	conda create -n pyrsd_emu python=3.9 pip numpy scipy
	
	# Activate the environment before using pip to install JAX
	conda activate pyrsd_emu
	
	# Install the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
	pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html


Then, pip/conda install the other required packages, including

	matplotlib
	pytorch
	pip install pytorch-lightning
	pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
	...

Install the pyrsd_emu package
	
	# from <current_dir>/pyrsd-emulator/ run:
	pip install -e .

IMPORTANT: This conda environment will run fine in say ipython, but to get the `posterior_analysis.ipynb` notebook running you need to follow the instructions here <https://docs.nersc.gov/services/jupyter/>, INCLUDING THE "Customizing Kernels with a Helper Shell Script" PORTION: 

First, add kernel to jupyter (access at https://jupyter.nersc.gov/hub/home)

	python -m ipykernel install --user --name pyrsd_emu --display-name pyRSD_emulator


Then, modify e.g. /global/homes/g/gstein/.local/share/jupyter/kernels/pyrsd_emu/kernel.json to contain the following

	{
	 "argv": [
	  "{resource_dir}/kernel-helper.sh",
	  "/global/homes/g/gstein/.conda/envs/pyrsd_emu/bin/python",
	  "-m",
	  "ipykernel_launcher",
	  "-f",
	  "{connection_file}"
	 ],
	 "display_name": "pyRSD_emulator",
	 "language": "python",
	 "metadata": {
	  "debugger": true
	 }
	}

Finally, create a kernel-helper.sh script in the same directory as where the kernel.json file is found. The kernel-helper.sh script should be made executable (chmod u+x kernel-helper.sh).

	#!/bin/bash                                                                                                          
	module load cudatoolkit/11.5
	module load cudnn/8.3.2
	module load python
	exec "$@"