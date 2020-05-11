import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST,ImageNet, CIFAR10

from neuralteleportation.layers.layer_utils import patch_module
from neuralteleportation.training import train
from neuralteleportation.models.test_models.dense_models import DenseNet as MNIST_DenseNet
from neuralteleportation.models.model_zoo.densenet import DenseNet
from neuralteleportation.models.model_zoo.resnet import *
from neuralteleportation.models.model_zoo.vgg import *
from neuralteleportation.models.model_zoo.unet import *
from neuralteleportation.layerhook import LayerHook
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

from tqdm import tqdm

import argparse

def argument_parser():
    '''
    simple argument parser for the experiement.

    Ex: python layerhooking.py --batch_size 10 --epochs 5 --model densenet
    '''
    parser = argparse.ArgumentParser(description='Simple argument parser for the layer hook experiment.')
    parser.add_argument("--batch_size",type=int, default=100)
    parser.add_argument("--epochs",type=int, default=5)
    parser.add_argument("--model",type=str,default="mnist_densenet",choices=['mnist_densenet',
                                                                            'densenet',
                                                                            'resnet',
                                                                            'vggnet',
                                                                            'unet',])
    parser.add_argument("--dataset",type=str,default="cifar10",choices=["mnist",
                                                                     "imagenet",
                                                                     "cifar10"])
    return parser.parse_args()


if __name__ == "__main__":
    
    # Hook Callback example, it can be anything.
    def hookCallback(self, input, output):
        '''
        small callback that prints the first outputed image to a pyplot figure.
        '''
        np_out = output.detach().cpu().numpy()
        plt.figure()
        plt.imshow(np_out[0,0,:,:])
        plt.colorbar()
    
    argparse = argument_parser()

    batch_size = argparse.batch_size
    epochs = argparse.epochs

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_train = None
    dataset_test = None
    num_of_classes = None
    transform = transforms.ToTensor()
    if argparse.dataset == "mnist":
        dataset_train = MNIST('/tmp', train=True, download=True, transform=transform)
        dataset_test = MNIST('/tmp', train=False, download=True, transform=transform)
        num_of_classes = 10
    elif argparse.dataset == "cifar10":
        dataset_train = CIFAR10('/tmp', train=True, download=True, transform=transform)
        dataset_test = CIFAR10('/tmp', train=False, download=True, transform=transform)
        num_of_classes = 10

    # Get the width and height of the image, then get the dimension of pixels values
    w,h = dataset_test.data.shape[1:3]
    dims = 1 if len(dataset_test.data.shape) < 4 else dataset_test.data.shape[3]

    test_img = torch.as_tensor(dataset_test[0][0])
    test_img = torch.unsqueeze(test_img,0)
    test_img = test_img.to(device=device)

    data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = None
    if argparse.model=='mnist_densenet':
        model = MNIST_DenseNet(in_channels=dims)
    elif argparse.model=='densenet':
        model = DenseNet()
    elif argparse.model=='resnet':
        model = ResNet()
    elif argparse.model=='vggnet':
        model = VGG()
    elif argparse.model=='unet':
        model = UNet()

    model = model.to(device=device)
    model = NeuralTeleportationModel(network=model, input_shape=(batch_size,dims,w,h))

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
        ones = torch.ones((1,dims,w,h)).to(device)
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
        print("Diff of x: image: ", img_diff.cpu().numpy())
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