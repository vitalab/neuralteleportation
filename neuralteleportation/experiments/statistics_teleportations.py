from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.models.model_zoo.vggcob import vgg16COB
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import save


def plot_angle_teleported_gradient(model, loss_func, input_shape=(4, 3, 32, 32), n_iter=200):
    """
    This method plots a histogram of the angles between the gradient of the network model and the gradients of
     n_iter teleportations of it.

    Args:
        model : NeuralTeleportationModel to test.

        input_shape : The shape of the input.

        loss_func : Loss function to compute gradients.

        n_iter : Number of teleportations computed for each cob_range.
    """
    x = torch.rand(input_shape, dtype=torch.float)
    y = torch.randint(low=0, high=9, size=(4,))

    original_weights = model.get_weights()
    grad1 = model.get_grad(x, y, loss_func, zero_grad=False)
    angle_results = []

    for _ in range(n_iter):
        model.set_weights(original_weights)
        model.random_teleport(cob_range=100, sampling_type='positive')
        grad2 = model.get_grad(x, y, loss_func, zero_grad=False)
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
    plt.title('Histogram of angles of gradient with multiple teleported gradients')
    plt.xlabel('Angle')
    plt.xlim(x_min, x_max)
    plt.show()


def plot_difference_teleported_gradients(model, loss_func, input_shape=(4, 3, 32, 32), n_iter=20, save_to_files=False):
    """
    This method plots the difference of the gradient of model and the gradient of a teleportation, by increasing the
    cob_range from 0.1 to 0.9 in n_iter iterations for usual cob_sampling. Each gradient is normalized by the norm of
    the weights producing the corresponding gradient.

    Args:
        model : NeuralTeleportationModel to test.

        input_shape : The shape of the input.

        loss_func : Loss function to compute gradients.

        n_iter : Number of teleportations computed for each cob_range.

        save_to_files : Flag to decide when to save results for the plot to a file.
    """
    x = torch.rand(input_shape, dtype=torch.float)
    y = torch.randint(low=0, high=9, size=(4,))

    original_weights = model.get_weights().detach().numpy()
    original_grad = model.get_grad(x, y, loss_func, zero_grad=False).numpy()
    original_grad = original_grad / np.linalg.norm(original_weights)

    differences = []
    variance = []
    x_axis = np.linspace(0.1, 0.9, n_iter)

    for i in range(n_iter):
        to_compute_mean = []
        for j in range(100):
            model.set_weights(original_weights)
            model.random_teleport(cob_range=x_axis[i], sampling_type='usual')

            teleported_weights = model.get_weights().detach().numpy()
            teleported_grad = model.get_grad(x, y, loss_func, zero_grad=False).numpy()
            teleported_grad = teleported_grad / np.linalg.norm(teleported_weights)

            diff = abs(np.linalg.norm(original_grad)-np.linalg.norm(teleported_grad))
            to_compute_mean.append(diff)

        variance.append(np.std(to_compute_mean))
        differences.append(np.mean(to_compute_mean))

    variance = np.array(variance)
    differences = np.array(differences)
    x_axis = np.array(x_axis)

    if save_to_files:
        save('variance.npy', variance)
        save('differences.npy', differences)
        save('x_axis.npy', x_axis)

    plt.errorbar(x_axis, differences, yerr=variance)
    plt.plot(x_axis, differences)

    plt.title('Difference of normalized magnitude between teleportations of the gradient')
    plt.ylabel('| ||Grad||/||W|| - ||Tel.Grad||/||Tel.W|| |')
    plt.xlabel('cob_range')
    plt.show()


if __name__ == '__main__':
    import torch.nn as nn

    input_shape = (4, 3, 32, 32)
    loss = nn.CrossEntropyLoss()

    network = vgg16COB(pretrained=False, num_classes=10, input_channels=3)
    network = NeuralTeleportationModel(network, input_shape=input_shape)

    plot_difference_teleported_gradients(network, loss)
    plot_angle_teleported_gradient(network, loss)
