import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from neuralteleportation.layer_utils import patch_module
from neuralteleportation.training import train
from neuralteleportation.models.test_models.dense_models import DenseNet as MNIST_DenseNet
from neuralteleportation.models.model_zoo.densenet import DenseNet
from neuralteleportation.models.model_zoo.resnet import *
from neuralteleportation.models.model_zoo.vgg import *
from neuralteleportation.models.model_zoo.unet import *
from neuralteleportation.layerhook import LayerHook
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

from tqdm import tqdm

if __name__ == "__main__":
    
    # Hook Callback example, it can be anything.
    def hookCallback(self, input, output):
        np_out = output.detach().cpu().numpy()
        plt.figure()
        plt.imshow(np_out[0,0,:,:])
        plt.colorbar()
    
    batch_size = 100
    epochs = 5

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = transforms.ToTensor()
    dataset_train = MNIST('/tmp', train=True, download=True, transform=transform)
    dataset_test = MNIST('/tmp', train=False, download=True, transform=transform)

    test_img = torch.as_tensor(dataset_test[0][0])
    test_img = torch.unsqueeze(test_img,0)
    test_img = test_img.to(device=device)

    data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # Create a model and train it.
    model = MNIST_DenseNet()
    model = model.to(device=device)
    model = NeuralTeleportationModel(network=model, input_shape=(batch_size,1,28,28), device=device)

    # Train the model.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for e in range(epochs):
        t = tqdm(range(len(data_loader)))
        for i, d in enumerate(data_loader,0):
            inputs, targets = d
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            
            optimizer.zero_grad()

            loss = criterion(output,targets)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item(),epochs=e+1)
            t.update()
        t.close()
    
    # Attach the hook the the
    hook = LayerHook(model)
    hook.set_hooked_layer("conv1", hookCallback)
    
    # We do a forward propagation to illustrate the example.
    with torch.no_grad():
        ones = torch.ones((1,1,28,28)).to(device)
        ones_output = model(ones)
        img_output = model(test_img)
                
        model.random_teleport()
        teleported_ones_output = model(ones)
        teleported_img_output = model(test_img)

        ones_diff = torch.abs(ones_output - teleported_ones_output)
        img_diff = torch.abs(img_output - teleported_img_output)

        print()
        print("=========Prediction Differences=========")
        print("Diff of x: torch.ones(): ",ones_diff.cpu().numpy())
        print("Diff of x: image of a 7: ", img_diff.cpu().numpy())
        print()

        ones_prediction = torch.max(ones_output, 1)
        teleported_ones_prediction = torch.max(teleported_ones_output, 1)
        img_prediction = torch.max(img_output, 1)
        teleported_img_prediction = torch.max(teleported_img_output, 1)

        print("=========Ones Prediction=========")
        print(ones_prediction)
        print(teleported_ones_prediction)
        print()
        print("=========Image Prediction=========")
        print(img_prediction)
        print(teleported_img_prediction)
        
    plt.show()