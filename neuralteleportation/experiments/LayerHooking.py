import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, ImageNet, CIFAR10

from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules as patch_module
from neuralteleportation.training.training import train
from neuralteleportation.training.config import *
from neuralteleportation.training import experiment_setup, experiment_run
from neuralteleportation.metrics import accuracy

from neuralteleportation.models.generic_models.dense_models import DenseNet as MNIST_DenseNet
from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.models.model_zoo.vggcob import vgg11COB
from neuralteleportation.layerhook import LayerHook
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

import argparse


def argument_parser():
    '''
    simple argument parser for the experiement.

    Ex: python layerhooking.py --batch_size 10 --epochs 5 --model resenet --layer_name layer1

    '''
    parser = argparse.ArgumentParser(
        description='Simple argument parser for the layer hook experiment.')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet", choices=['mnist_densenet',
                                                                        'densenet',
                                                                        'resnet',
                                                                        'vggnet'])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--layer_name", type=str, default="conv1")
    parser.add_argument("--cob_sampling", type=str, defaut="usual")
    return parser.parse_args()


if __name__ == "__main__":
    # Hook Callback example, it can be anything you want.
    def hookCallback(self, input, output):
        '''
        small callback that prints the first outputed image to a pyplot figure.
        '''
        np_out = output.detach().cpu().numpy()
        plt.figure()
        plt.imshow(np_out[0, 0, :, :])
        plt.colorbar()

    args = argument_parser()
    batch_size = args.batch_size

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = None
    transform = transforms.ToTensor()
    num_classes = 10

    if args.model == 'mnist_densenet':
        model = MNIST_DenseNet(in_channels=1)
    elif args.model == 'densenet':
        model = densenet121COB(num_classes=num_classes)
    elif args.model == 'resnet':
        model = resnet18COB(input_channels=3, num_classes=num_classes)
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    elif args.model == 'vggnet':
        model = vgg11COB(num_classes=num_classes)

    if args.dataset == "mnist":
        trainset, valset, testset = experiment_setup.get_mnist_datasets()
    elif args.dataset == "cifar10":
        trainset, valset, testset = experiment_setup.get_cifar10_datasets()

    data_train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Get the width and height of the image, then get the dimension of pixels
    # values
    w, h = testset.transform.transforms[0].size if isinstance(testset.transform, transforms.Compose) else testset.data.shape[1:3]
    dims = 1 if len(testset.data.shape) < 4 else testset.data.shape[3]

    test_img = torch.as_tensor(testset[0][0])
    test_img = torch.unsqueeze(test_img, 0)
    test_img = test_img.to(device=device)

    # Change the model to a teleportable model.
    model = model.to(device=device)
    model = NeuralTeleportationModel(network=model, input_shape=(batch_size, dims, w, h))

    metric = TrainingMetrics(torch.nn.CrossEntropyLoss(), [accuracy])
    config = TrainingConfig(epochs=args.epochs, device=device, batch_size=batch_size)
    train(model, train_dataset=trainset, metrics=metric, config=config)

    # Attach the hook to a specific layer
    hook_state = (args.layer_name, hookCallback)
    hook = LayerHook(model, hook_state)

    # We do a forward propagation to illustrate the example.
    model.eval()
    ones = torch.ones((1, dims, w, h)).to(device)
    ones_output = model(ones)
    img_output = model(test_img)

    if isinstance(model, NeuralTeleportationModel):
        model.random_teleport()
    teleported_ones_output = model(ones)
    teleported_img_output = model(test_img)

    img_prediction = torch.max(img_output, 1)
    teleported_img_prediction = torch.max(teleported_img_output, 1)

    print("=========Image Prediction=========")
    print(img_prediction)
    print(teleported_img_prediction)

    plt.show()
