# Neural Teleportation    
 
## Description   

Neural network teleportation using representation theory of quivers. 

## How to run   
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
from neuralteleportation import Whatever   
``` 

## Repository content

This repository contains the code necessary to teleport a neural network. 

* [neuralteleportation](neuralteleportation) : contains the main classes for network teleportation. 
* [layers](neuralteleportation/layers): contains the classes necessary for teleporting individual layers. 
* [models](neuralteleportation/models): contains frequently used models such as Resnet, VGG...
* [experiments](neuralteleportation/experiments): contains experiments using teleportation. 
* [tests](tests): contains black-box tests for network teleportation. 

## Running experiments

* Micro-teleportations (Figure 4): Running the script *neuralteleportation/experiments/micro_teleportation/microteleportation.py* produces histograms of angles of gradient vs micro-teleportations for MLP, VGG, ResNet and DenseNet on CIFAR-10 and random data.
* Interpolations (Figure 5): 
* Teleportation vs SGD (Figure 6): Running the yaml file *neuralteleportation/experiments/config/SGD_vs_teleport.yml*
* Gradient changed by teleportation (Figure 7): Running the script *neuralteleportation/utils/statistics_teleportations.py* produces one plot for each model, MLP, VGG, ResNet and DenseNet on CIFAR-10
* Teleportation vs SGD on other activation functions (Figure 8): Running the yaml file *neuralteleportation/experiments/config/OtherActivations.yml*
* Teleportation vs different initializations (Figure 9): Running the yaml file *neuralteleportation/experiments/config/Initializations.yml*
* Pseudo-teleportation vs SGD (Figure 10): Running the yaml file *neuralteleportation/experiments/config/pseudoo_teleportation.yaml*
* Histogram comparision before and after teleportation (Figure 11): Running the script *neuralteleportation/experiments/weights_histogram.py*

## Known Limitations

* Can't use opperations in the foward method (only nn.Modules)
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
