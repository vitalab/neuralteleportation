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

Reproducing all our results requires about 160 GPU\*hours. We also give estimates
per experiments, below. A "GPU" corresponds to an Nvidia V100 GPU, 8 cores on an
Intel Gold 6148 Skylake GPU, and 32GB of RAM.  

### Convergence boost of teleportation (Figure 6)

To run with on a single GPU:

```bash
python neuralteleportation/experiments/teleport_training.py \
  neuralteleportation/experiments/config/SGD_vs_teleport.yml \ 
  --out_root_dir ./out_fig6
```

The output folder will contain the results necessary to produce the plots. See section "Generating box plots from the metrics files of training experiments" below.

**The compute required for this experiment is about 40 GPU\*hours**. If you have access to a computer with multiple
GPUs, you can parallelize as follows:

```bash
# Generate a YAML config file for each training job
python neuralteleportation/experiments/config/unravel_matrix_config.py \
  neuralteleportation/experiments/config/SGD_vs_teleport.yml \
  --output_dir ./configs_fig6

# Auto-detect the number of GPUs
cvd_tokens=$( echo $CUDA_VISIBLE_DEVICES | tr ',' ' '  )
cvd_array=( $cvd_tokens )
N_GPUS=${#cvd_array[@]}

# Run N_GPUS training jobs in parallel using GNU Parallel
ls ./configs_fig6/*.yml | parallel -j $N_GPUS --linebuffer \
  CUDA_VISIBLE_DEVICES='$(({%} - 1))' \
  python neuralteleportation/experiments/teleport_training.py {} \
  --out_root_dir ./out_fig6
```

If you have access to resources on a cluster through SLURM, the following will
automatically submit a SLURM job for each training job:

```bash
bash neuralteleportation/experiments/submit_teleport_training_batch.sh \
  -f neuralteleportation/experiments/config/SGD_vs_teleport.yml \  
  --out_root_dir <OUTPUT_DIR> \
  -p <THIS_REPO> \
  -d <DATASETS_FOLDER> \
  -v <PYTHON_ENV_FOLDER> \
  -m <EMAIL_ADDRESS>
```

Please refer to the script for more details.

### Teleportation with other activation functions (Figure 8)

Use the instructions in section "Convergence boost of teleportation" above, but with
this config file: `neuralteleportation/experiments/config/OtherActivations.yml`. 

This experiment requires about 32 GPU*hours (probably less).

### Teleportation with different initializations (Figure 9)

Use the instructions in section "Convergence boost of teleportation" above, but with
this config file: `neuralteleportation/experiments/config/Teleportation_vs_Initializers.yml`. 

**WARNING** The MLP model with xavier init needs a gain=1.0 to converge, unlike the other models where the default gain=0.02 is enough.

This experiment requires about 40 GPU*hours.

### Pseudo-teleportation (Figure 10)

Use the instructions in section "Convergence boost of teleportation" above, but with
this config file: `neuralteleportation/experiments/config/SGD_vs_PseudoTeleport.yml` .

This experiment requires about 40 GPU*hours.

### Micro-teleportations (Figure 4)

Running the script *neuralteleportation/experiments/micro_teleportation/microteleportation.py*

```bash
python neuralteleportation/experiments/micro_teleportation/micro_teleportation.py
```

produces histograms of angles of gradient vs micro-teleportations for MLP and VGG on CIFAR-10 and random data.

This experiment requires in the order of 1 GPU*hour.

### Interpolations for flatness visualization (Figure 5)

Running the script *neuralteleportation/experiments/flatness_1D_interp.py*

```bash
python neuralteleportation/experiments/flatness_1D_interp.py
```

trains two MLPs A and B, with batch-sizes 8 and 2014, respectively. Then interpolates between the two trained models and plots the accuracy/loss profile in that interpolation. Finally, teleports A and B with CoB-range of 0.9, and plots the accuracy/loss profile of the interpolation between the teleportations of A and B.

Hyperparameters can be found in the usage of the script. 
**WARNING** It is probable that the MLPs give NaN's during training. If this happens, just re-run the script.

This experiment requires in the order of 1-2 GPU*hours.

### Gradient changed by teleportation (Figure 7)

Running the script *neuralteleportation/utils/statistics_teleportations.py*

```bash
python neuralteleportation/utils/statistics_teleportations.py
```

produces the four plots shown in figure 7 in the paper for MLP, VGG, ResNet and DenseNet on CIFAR-10.

This experiment requires in the order of 1 GPU*hour.

### Weight histogram comparison before and after teleportation (Figure 11)

Running the script *neuralteleportation/experiments/weights_histogram.py*

```bash
python neuralteleportation/experiments/weights_histogram.py
```

produces the histograms shown in figure 11 in the paper.

### Generating box plots from the metrics files of training experiments

The following can be used to generate the plots from figure 6, for example.
Replace `<EXPERIMENT_DIR>` with the `--out_root_dir` you have used
(see instructions above).

```bash
python neuralteleportation/experiments/visualize/generate_mean_graphs.py \
  --experiment_dir <EXPERIMENT_DIR> \
  --metrics val_accuracy \
  --group_by teleport optimizer \
  --boxplot --box_epochs 30 60 95 \ 
  --out_dir ./results
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
