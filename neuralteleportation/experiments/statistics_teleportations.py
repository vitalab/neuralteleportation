from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import train
from neuralteleportation.models.model_zoo.vggcob import vgg16COB
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_angle_teleported_gradient(model, loss_func, input_shape = (32, 3, 32, 32), n_iter=20):
    x = torch.rand(input_shape, dtype=torch.float).to(device='cuda')
    y = torch.randint(low=0, high=9, size=(32,)).to(device='cuda')
    grad1 = model.get_grad(x, y, loss_func, zero_grad=False).cpu()
    angle_results = []

    for _ in range(n_iter):
        model.random_teleport(cob_range=50, sampling_type='positive')
        grad2 = model.get_grad(x, y, loss_func, zero_grad=False).cpu()
        dot_prod = np.longfloat(np.dot(grad1, grad2) /
                                (np.linalg.norm(grad1) * np.linalg.norm(grad2)))
        angle = np.degrees(np.arccos(dot_prod))
        angle_results.append(angle)

    angle_results = np.array(angle_results)

    delta = np.maximum(1.0, angle_results.std() * 3)
    x_min = 90 - delta
    x_max = 90 + delta

    bin_height, bin_boundary = np.histogram(np.array(angle_results))
    width = bin_boundary[1] - bin_boundary[0]
    bin_height = bin_height / float(max(bin_height))
    plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.05))
    plt.title('Angle of gradients between teleportations')
    plt.xlim(x_min, x_max)
    plt.show()


def plot_magnitude_teleported_gradient(model, loss_func, input_shape = (32, 3, 32, 32), n_iter=20):
    x = torch.rand(input_shape, dtype=torch.float).to(device='cuda')
    y = torch.randint(low=0, high=9, size=(32,)).to(device='cuda')
    grad1 = model.get_grad(x, y, loss_func, zero_grad=False).cpu()
    weights1 = model.get_weights().cpu().detach().numpy()
    grad1 = grad1 / np.linalg.norm(weights1)
    differences = []

    for _ in range(n_iter):
        model.random_teleport(cob_range=50, sampling_type='positive')
        grad2 = model.get_grad(x, y, loss_func, zero_grad=False).cpu()
        weights2 = model.get_weights().cpu().detach().numpy()
        grad2 = grad2 / np.linalg.norm(weights2)
        diff = np.linalg.norm(grad1)-np.linalg.norm(grad2)

        differences.append(diff)

    differences = np.array(differences)

    plt.plot(differences)
    plt.title('Difference of normalized magnitude between teleportations of the gradient')
    #plt.xlim(x_min, x_max)
    plt.show()


def plot_grad_over_weights(model, loss_func, input_shape = (32, 3, 32, 32), n_iter=30):
    x = torch.rand(input_shape, dtype=torch.float).to(device='cuda')
    y = torch.randint(low=0, high=9, size=(32,)).to(device='cuda')

    cob_range = 0.2
    result = []

    for _ in range(n_iter):
        cob_range += 0.1
        same_cob_ratios = []

        for _ in range(20):
            model.random_teleport(cob_range=cob_range, sampling_type='positive')
            weights = model.get_weights().cpu().detach().numpy()
            #weights = weights/np.linalg.norm(weights)
            #model.set_weights(weights)
            grad = model.get_grad(x, y, loss_func, zero_grad=False).cpu()
            z =  np.linalg.norm(grad)/ np.linalg.norm(weights)
            print(np.linalg.norm(grad))
            print(np.linalg.norm(weights))
            print('------')
            same_cob_ratios.append(z)

        ratio = np.mean(same_cob_ratios)
        result.append(ratio)

    result = np.array(result)

    plt.title('vgg16COB, cob_range vs ||Grad||/||W||')
    plt.xlim(0, n_iter*0.1 + 1)
    plt.ylim(np.amin(result), np.amax(result))
    plt.plot(result)
    plt.show()


if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    import torch.nn as nn
    from torch.utils.data import DataLoader

    input_shape = (32, 3, 32, 32)
    loss_func = nn.CrossEntropyLoss()

    #CIFAR10_train = CIFAR10('/tmp', train=True, download=True, transform=transforms.ToTensor())
    #CIFAR10_val = CIFAR10('/tmp', train=False, download=True, transform=transforms.ToTensor())
    #CIFAR10_test = CIFAR10('/tmp', train=False, download=True, transform=transforms.ToTensor())

    model = vgg16COB(pretrained=False, num_classes=10, input_channels=3)
    model = NeuralTeleportationModel(model, input_shape=input_shape).to(device='cuda')

    #config = TrainingConfig(device='cuda', epochs=5, lr=0.1)
    #metrics = TrainingMetrics(loss_func, [accuracy])

    #train_loader = DataLoader(CIFAR10_train, batch_size=config.batch_size)

    #train(model.to(device='cuda'), train_dataset=CIFAR10_train, metrics=metrics, config=config,
    #      val_dataset=CIFAR10_val)
    #plot_grad_over_weights(model, loss_func)
    plot_magnitude_teleported_gradient(model, loss_func)

