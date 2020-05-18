# How to Teleport

To teleport a Pytorch neural network (nn.Module) one must use the 
[NeuralTeleportationModel](neuralteleportationmodel.py) class and use layers that inherit from 
[NeuralTeleportationLayerMixin](layers/neuralteleportationlayers.py). The logic for teleportation is mainly split into 
these two files. 

**One must respect the know limitations in this [README](../README.md).**

To teleport, one has two options. 

1. If the model is a simple (no residual connections or skip connections), one define a model with standard 
nn.Modules and use the  [```swap_model_modules_for_COB_modules()```](layers/layer_utils.py). 
```python
model = torch.nn.Sequential(
    nn.Conv2d(1, 32, 3, 1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, stride=2),
    nn.ReLU(),
    Flatten(),
    nn.Linear(9216, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

model = swap_model_modules_for_COB_modules(model)

model = NeuralTeleportationModel(model, input_shape)
``` 

2. If the model has skip or residual connections, the model must be constructed using the COB layers defined in [layers](layers). 
```python
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3)
        self.conv4 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3)
       
        self.relu2 = ReLUCOB()
        self.relu3 = ReLUCOB()
        self.relu4 = ReLUCOB()
        self.add = Add()
        self.flatten = FlattenCOB()
        self.fc1 = LinearCOB(3 * 24 * 24, 10)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))

        x2 = self.add(x1, x2)  # equivalent to x2 += x1

        x3 = self.relu3(self.conv3(x2))
        x4 = self.conv4(x3)
        x = self.flatten(x4)
        x = self.relu4(self.fc1(x))
        return x

model =  ResidualNet()

model = NeuralTeleportationModel(model, input_shape)
``` 


## Model zoo 

The model zoo contains well-known models that we implemented with the teleportation framework. 
The implementations come in part from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)

```python
from neuralteleportation.models.model_zoo.resnetcob import resnet50COB

model =  resnet50COB()
model = NeuralTeleportationModel(model, input_shape)
``` 
