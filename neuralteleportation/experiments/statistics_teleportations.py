from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import train
from neuralteleportation.models.model_zoo.vggcob import vgg16COB
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_angle_teleported_gradient(model, loss_func, input_shape = (4, 3, 32, 32), n_iter=1000):
    x = torch.rand(input_shape, dtype=torch.float)
    y = torch.randint(low=0, high=9, size=(4,))
    original_weights = model.get_weights()
    original_grad = model.get_grad(x, y, loss_func, zero_grad=False)
    angle_results = []

    for _ in range(n_iter):
        model.set_weights(original_weights)
        model.random_teleport(cob_range=50, sampling_type='positive')
        teleported_grad = model.get_grad(x, y, loss_func, zero_grad=False)
        dot_prod = np.longfloat(np.dot(original_grad, teleported_grad) /
                                (np.linalg.norm(original_grad) * np.linalg.norm(teleported_grad)))
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
    plt.ylabel('Count')
    plt.xlabel('Angle')
    plt.xlim(x_min, x_max)
    plt.show()


def plot_magnitude_teleported_gradient(model, loss_func, input_shape = (4, 3, 32, 32), n_iter=20):
    x = torch.rand(input_shape, dtype=torch.float)
    y = torch.randint(low=0, high=9, size=(4,))
    original_weights = model.get_weights().detach().numpy()
    original_grad = model.get_grad(x, y, loss_func, zero_grad=False).numpy()
    original_grad = original_grad / np.linalg.norm(original_weights)
    differences = []
    variance = []
    center = 0.2
    xaxis = []

    for i in range(n_iter):
        center += 0.05
        xaxis.append(center)
        to_compute_mean = []
        for j in range(10):
            model.set_weights(original_weights)
            model.random_teleport(cob_range=0.001, sampling_type='positive_centered', center=center)

            teleported_grad = model.get_grad(x, y, loss_func, zero_grad=False).numpy()
            teleported_weights = model.get_weights().detach().numpy()
            teleported_grad = teleported_grad / np.linalg.norm(teleported_weights)
            diff = abs(np.linalg.norm(original_grad)-np.linalg.norm(teleported_grad))

            to_compute_mean.append(diff)

        variance.append(np.var(to_compute_mean))
        differences.append(np.mean(to_compute_mean))

    variance = np.array(variance)
    differences = np.array(differences)
    xaxis = np.array(xaxis)
    plt.errorbar(xaxis, differences, yerr=variance)
    plt.plot(xaxis, differences)
    plt.title('Difference of normalized magnitude between teleportations of the gradient')
    #plt.xlim(0, n_iter)
    plt.ylabel('| ||Grad||/||W|| - ||Tel.Grad||/||Tel.W|| |')
    plt.xlabel('cob_range in (x-0.1,x+0.1)')
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
    model = NeuralTeleportationModel(model, input_shape=input_shape)

    #config = TrainingConfig(device='cuda', epochs=5, lr=0.1)
    #metrics = TrainingMetrics(loss_func, [accuracy])

    #train_loader = DataLoader(CIFAR10_train, batch_size=config.batch_size)

    #train(model.to(device='cuda'), train_dataset=CIFAR10_train, metrics=metrics, config=config,
    #      val_dataset=CIFAR10_val)
    plot_magnitude_teleported_gradient(model, loss_func)
    #plot_angle_teleported_gradient(model, loss_func)

