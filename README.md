# Neural Teleportation    
 
Neural network teleportation using representation theory of quivers. 

**NOTE TO REVIEWERS: The instructions for reproducing our results are in the section
"Running experiments", below.**

## Repository content

This repository contains the code necessary to teleport a neural network. 

* [neuralteleportation](neuralteleportation) : contains the main classes for network teleportation. 
* [layers](neuralteleportation/layers): contains the classes necessary for teleporting individual layers. 
* [models](neuralteleportation/models): contains frequently used models such as MLP, VGG, ResNet and DenseNet.
* [experiments](neuralteleportation/experiments): contains experiments using teleportation. 
* [tests](tests): contains black-box tests for network teleportation. 

## Setting up 
First, install dependencies   
```bash
# set-up project's environment
cd neuralteleportation
conda env create -f neuralteleportation.yml

# activate the environment so that we install the project's package in it
conda activate neuralteleportation
pip install -e .

```
To test that the project was installed successfully, you can try the following command from the Python REPL:
```python

# now you can do:
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel   
``` 

## Running experiments

The following instructions allow to reproduce the results found in our paper.

Reproducing all our results requires XXX GPU\*hours. We also give estimates per
experiments, below. Our estimates are upper bounds. A "GPU" corresponds to an Nvidia
V100 GPU, 8 cores on a Intel Gold 6148 Skylake GPU, and 32GB of RAM.  

### Convergence boost of teleportation (Figure 6)

Running the yaml file in a cluster with slurm *neuralteleportation/experiments/config/SGD_vs_teleport.yml* 

```bash
$HOME/neuralteleportation/neuralteleportation/experiments/submit_teleport_training_batch.sh -p $HOME/neuralteleportation/ -d $HOME/datasets/ -f $HOME/neuralteleportation/neuralteleportation/experiments/config/SGD_vs_teleport.yml -v $HOME/virtualenv/ -m email@email.email --out_root_dir $HOME/scratch/SGDvsTeleport/Metrics/VGG_cifar10/
```
The metrics have to be in a directory specific to the model and the dataset. For example, a directory Metrics/VGG_cifar10, should contain 5 runs over two optimizers (SGD and SGD+Momentum), three learning rates (0.01, 0.001 and 0.0001) done with and without teleportation (60 experiments total) on the VGG model for the dataset CIFAR-10.

### Teleportation with other activation functions (Figure 8)

Running the yaml file in a cluster with slurm *neuralteleportation/experiments/config/OtherActivations.yml* 

```bash
$HOME/neuralteleportation/neuralteleportation/experiments/submit_teleport_training_batch.sh -p $HOME/neuralteleportation/ -d $HOME/datasets/ -f $HOME/neuralteleportation/neuralteleportation/experiments/config/OtherActivations.yml -v $HOME/virtualenv/ -m email@email.email --out_root_dir $HOME/scratch/OtherActivations/Metrics/tanh_cifar10
```

produces all the metrics needed to reproduce the plots of figure 8 in the paper. The metrics have to be in a directory specific to the activation and the dataset. For example, a directory Metrics/tanh_cifar10, should contain 5 runs over two optimizers (SGD and SGD+Momentum), three learning rates (0.01, 0.001 and 0.0001) done with and without teleportation (60 experiments total) on the MLP model with tanh activation for the dataset CIFAR-10.

### Teleportation with different initializations (Figure 9)

Running the yaml file in a cluster with slurm *neuralteleportation/experiments/config/Initializations.yml* 

```bash
$HOME/neuralteleportation/neuralteleportation/experiments/submit_teleport_training_batch.sh -p $HOME/neuralteleportation/ -d $HOME/datasets/ -f $HOME/neuralteleportation/neuralteleportation/experiments/config/Teleportation_vs_Initializers.yml -v $HOME/virtualenv/ -m email@email.email --out_root_dir $HOME/scratch/Initializations/Metrics/xavier_VGG_cifar10
```

produces all the metrics needed to reproduce the plots of figure 9 in the paper. The metrics have to be in a directory specific to the initialization, the model and the dataset. For example, a directory Metrics/xavier_VGG_cifar10, should contain 5 runs over two optimizers (SGD and SGD+Momentum), three learning rates (0.01, 0.001 and 0.0001) done with and without teleportation (60 experiments total) on the VGG model with xavier initialization for the dataset CIFAR-10. **WARNING** The MLP model with xavier init needs a gain=1.0 to converge, unlike the other models where the default gain=0.02 is enough.

### Pseudo-teleportation (Figure 10)

Running the yaml file in a cluster with slurm *neuralteleportation/experiments/config/pseudo_teleportation.yml* 

```bash
$HOME/neuralteleportation/neuralteleportation/experiments/submit_teleport_training_batch.sh -p $HOME/neuralteleportation/ -d $HOME/datasets/ -f $HOME/neuralteleportation/neuralteleportation/experiments/config/SGD_vs_PseudoTeleport.yml -v $HOME/virtualenv/ -m email@email.email --out_root_dir $HOME/scratch/Pseudo_teleport/VGG_cifar10
```

produces all the metrics needed to reproduce the plots of figure 10 in the paper. The metrics have to be in a directory specific to the activation and the dataset. For example, a directory VGG_cifar10, should contain 5 runs over two optimizers (SGD and SGD+Momentum), three learning rates (0.01, 0.001 and 0.0001) done with and without pseudo-teleportation (60 experiments total) on the VGG model for the dataset CIFAR-10.

### Micro-teleportations (Figure 4)

Running the script *neuralteleportation/experiments/micro_teleportation/microteleportation.py*

```bash
python neuralteleportation/experiments/micro_teleportation/micro_teleportation.py
```

produces histograms of angles of gradient vs micro-teleportations for MLP and VGG on CIFAR-10 and random data.


### Interpolations for flatness visualization (Figure 5)

Running the script *neuralteleportation/experiments/flatness_1D_interp.py*

```bash
python neuralteleportation/experiments/flatness_1D_interp.py
```

trains two MLPs A and B, with batch-sizes 8 and 2014, respectively. Then interpolates between the two trained models and plots the accuracy/loss profile in that interpolation. Finally, teleports A and B with CoB-range of 0.9, and plots the accuracy/loss profile of the interpolation between the teleportations of A and B.

Hyperparameters can be found in the usage of the script. 
**WARNING** It is probable that the MLPs give NaN's during training. If this happens, just re-run the script.

### Gradient changed by teleportation (Figure 7)

Running the script *neuralteleportation/utils/statistics_teleportations.py*

```bash
python neuralteleportation/utils/statistics_teleportations.py
```

produces the four plots shown in figure 7 in the paper for MLP, VGG, ResNet and DenseNet on CIFAR-10.

### Weight histogram comparison before and after teleportation (Figure 11)

Running the script *neuralteleportation/experiments/weights_histogram.py*

```bash
python neuralteleportation/experiments/weights_histogram.py
```

produces the histograms shown in figure 11 in the paper.

### Generating box plots from the metrics files of training experiments

Metrics have to be ordered by model and dataset, as described in each of the experiments shown above.

```bash
python neuralteleportation/experiments/visualize/generate_mean_graphs.py --metrics validate_accuracy --group_by teleport optimizer --experiment_dir ../Results_NeuralTeleportation/SGDvsTeleport/Metrics/VGG_cifar10/ --boxplot --box_epochs 30 60 95 --out_dir ../Results_NeuralTeleportation/SGDvsTeleport/Plots/
```

## Known Limitations

* Can't use operations in the forward method (only nn.Modules)
* Can't use nn.modules more than once (causes error in graph creation and if the layer have teleportation parameters)
* The order of layers is important when using Skip connections and residual connections. 
The first input must be computed in the network before the second input. The following example illustrates how to use these layers.
```python
class network(nn.Module):
    def forward(x):
        x1 = layer1(x) # Computed first
        x2 = layer2(x1) # Computed second

        x3 = Add(x1, x2) # x1 comes before x2.
``` 
