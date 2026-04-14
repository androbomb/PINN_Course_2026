# Hands On material for the PhD course "Physics Informed Neural Network" AA2025/26 

## Prerequisites: how to install PhysicsNemo-SYM

### Using Apptainer/Singularity from Docker
NVIDIA offers `PhysicsNemo-SYM` in a pre-defined container in their [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo):
```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
```

This means that, if you have [`apptainer`](https://apptainer.org/) installed (on how to install Apptainer, see [their guide here](https://github.com/apptainer/apptainer/blob/main/INSTALL.md)).

The `physicsnemo2506.def` here can be built
```bash
apptainer build physicsnemo2506_container.sif physicsnemo2506.def
```
And it can either be run interactively
```bash
apptainer run physicsnemo2506_container.sif
```
or added as a kernel to Jupyter in your local space; to do so, we need to create the folder
```bash
mkdir -p ~/.local/share/jupyter/kernels/physicsnemo2506
```
and then create the `kernel.json` file (e.g., via `vim` or `nano`)
```
{
    "display_name": "PhysicsNemo 2506",
    "argv": [
        "/usr/bin/apptainer",
        "run",
        "--nv",
        "--bind",
        "/home",
        "/path/to/physicsnemo2506_container.sif",
        "python3",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}"
    ],
    "language": "python",
    "metadata": {
        "debugger": true
    },
    "env": {
        "LD_LIBRARY_PATH": ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn"
    }
}

```
Notice that we added in the env section the `LD_LIBRARY_PATH` env var, adding to the original one the path to cudnn as `/usr/local/lib/python3.12/dist-packages/nvidia/cudnn`.

### Using Conda

It is kinda difficult to get the installation with conda right. 

We need
```bash
# 1. Create env and add torch
# Create a new environment (Python 3.10 is a safe, compatible choice)
conda create -n physicsnemo_env python=3.10 -y
# Activate the environment
conda activate physicsnemo_env
# Install cuda toolkit (e.g., 12.6)
conda install -c nvidia cuda-toolkit=12.6 -y
# Install PyTorch with a compatible CUDA version 
# but from pip since torch does not longer supports conda
# please notice the index url pointing to cu126 as above
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# [Optional] Check that nvcc is from Conda and is version 12.6
nvcc --version
# [Optional] check torch install 
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"

# 2. Install PhysicsNeMo
# install physicsnemo
pip install nvidia-physicsnemo
# Install dependencies for the symbolic part
pip install Cython
# Install PhysicsNeMo-Sym with build isolation disabled
pip install nvidia-physicsnemo-sym --no-build-isolation

# 3. [Optional] Create kernel
# Install ipykernel in the current environment
pip install ipykernel -y
# Register the kernel with Jupyter
python -m ipykernel install --user --name physicsnemo_env --display-name "PhysicsNeMo (conda)"

# 4. [Optional] Final checks
python -c "import physicsnemo; import physicsnemo.sym; print(physicsnemo.__version__); print(physicsnemo.sym.__version__)"
```

## Lectures

The course will be divided in _frontal lectures_ and _hands on lectures_; this repository contains the code for the hands on sessions. 
















