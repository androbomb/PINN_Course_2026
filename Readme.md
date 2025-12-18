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
apptainer run barumini_gnn.sif
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

## Lectures

The course will be divided in _frontal lectures_ and _hands on lectures_; this repository contains the code for the hands on sessions. 
















