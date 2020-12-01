import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_histogram_teleported_gradients(network, pbar, n_part=5, nb_batches=20, network_descriptor='',
                                        device='cpu') -> None:
    """
    This method computes an histogram of angles between the gradient of a network with gradients of teleportations
    of it while the change of basis increases and is centered around 1. We assume that network is a ReLU network
    or a teleportation of it (with any change of basis),and computes statistics within the same landscape
    by applying only positive changes of basis.

    Args:
        network :               NeuralTeleportationModel

        pbar :                  tqdm progress bar

        n_part:                 The number of partitions of the interval [0.001, 0.4] to sample the change of basis
                                range.

        nb_batches:             Number of batches from the data to compute histogram

        network_descriptor:     String describing the content of the network.

        device:                 Device for computations
    """
    original_weights = network.get_weights()

    loss_func = torch.nn.CrossEntropyLoss()

    # This measures the increase of the change of basis in each iteration
    cob = np.linspace(0.001, 0.99, n_part)

    for i in range(n_part):
        rand_angle_results = []
        rand_rand_angle_results = []
        grad_grad_angle_results = []

        for batch_idx, (data, target) in pbar:
            x, y = data.to(device), target.to(device)

            network.set_weights(original_weights)
            grad = network.get_grad(x, y, loss_func, zero_grad=False).detach().cpu().numpy()

            network.random_teleport(cob_range=cob[i])
            grad_tele = network.get_grad(x, y, loss_func, zero_grad=False).detach().cpu().numpy()

            random_vector = torch.rand(grad.shape, dtype=torch.float, device='cpu')-0.5
            random_vector2 = torch.rand(grad.shape, dtype=torch.float, device='cpu')-0.5

            # Normalized scalar product
            grad_grad_prod = np.longfloat(np.dot(grad, grad_tele) /
                                          (np.linalg.norm(grad)*np.linalg.norm(grad_tele)))
            grad_grad_angle = np.degrees(np.arccos(grad_grad_prod))

            rand_dot_prod = np.longfloat(np.dot(grad, random_vector) /
                                         (np.linalg.norm(grad)*np.linalg.norm(random_vector)))
            rand_angle = np.degrees(np.arccos(rand_dot_prod))

            rand_rand_dot_prod = np.longfloat(np.dot(random_vector2, random_vector) /
                                              (np.linalg.norm(random_vector2)*np.linalg.norm(random_vector)))
            rand_rand_angle = np.degrees(np.arccos(rand_rand_dot_prod))

            grad_grad_angle_results.append(grad_grad_angle)
            rand_angle_results.append(rand_angle)
            rand_rand_angle_results.append(rand_rand_angle)

            # Early stop at nb_batches, otherwise we get NaN's
            if (batch_idx+1) % nb_batches == 0:
                break

        grad_grad_angle_results = np.array(grad_grad_angle_results)
        rand_angle_results = np.array(rand_angle_results)
        rand_rand_angle_results = np.array(rand_rand_angle_results)

        # Limits to appreciate difference between the angles
        delta = np.maximum(1.0, rand_rand_angle_results.std() * 3)
        x_min = 90-delta
        x_max = 90+delta

        deltag = np.maximum(1.0, grad_grad_angle_results.std() * 3)
        meang = grad_grad_angle_results.mean()
        x_ming = meang - deltag
        x_maxg = meang + deltag

        bin_height, bin_boundary = np.histogram(np.array(grad_grad_angle_results))
        width = bin_boundary[1] - bin_boundary[0]
        bin_height = bin_height / float(max(bin_height))
        plt.subplot(3, 1, 1)
        plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='r')
        plt.title(f'{network_descriptor}: intra_landscape COB; range: {cob[i]}')
        plt.xlim(x_ming, x_maxg)
        plt.legend(['Gradient \n vs \n Teleported gradient'])

        bin_height, bin_boundary = np.histogram(np.array(rand_angle_results))
        width = bin_boundary[1] - bin_boundary[0]
        bin_height = bin_height / float(max(bin_height))
        plt.subplot(3, 1, 2)
        plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='b')
        plt.xlim(x_min, x_max)
        plt.legend(['Gradient\n vs \n Random Vector'])

        bin_height, bin_boundary = np.histogram(np.array(rand_rand_angle_results))
        width = bin_boundary[1] - bin_boundary[0]
        bin_height = bin_height / float(max(bin_height))
        plt.subplot(3, 1, 3)
        plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
        plt.xlim(x_min, x_max)
        plt.legend(['Random Vector\n vs \n Random Vector'])

        plt.xlabel('Angle in degrees')
        plt.show()


def plot_difference_teleported_gradients(network, pbar, nb_teleportations=10, n_part=15, network_descriptor='',
                                         device='cpu', limit_batches=10):
    """
    This method plots the difference of the gradient of the network and the gradient of a teleportation, by
    partitioning the cob_range from 0.1 to 0.9 in n_part parts for intra_landscape cob_sampling. Each gradient
    is normalized by the norm of the weights producing the corresponding gradient.

    Args:
        network:                NeuralTeleportationModel

        pbar:                   tqdm progress bar

        nb_teleportations:      Number of teleportations computed for each cob_range.

        n_part:                 The number of partitions of the interval [0.01, 0.9] to sample the change of basis
                                range.

        network_descriptor:     Name of the network for distinction.

        device:                 Device for computations.

        limit_batches:          Number of batches used to compute gradients for each CoB-range.
    """
    loss_func = torch.nn.CrossEntropyLoss()

    if device == 'cuda':
        original_weights_cuda = network.get_weights()

    original_weights = network.get_weights().detach().cpu().numpy()

    differences = []
    variance = []

    # Grid to sample the change of basis from
    x_axis = np.linspace(0.01, 0.9, n_part)

    for i in range(n_part):
        to_compute_mean = []

        for batch_idx, (data, target) in pbar:

            for j in range(nb_teleportations):
                if device == 'cuda':
                    network.set_weights(original_weights_cuda)
                else:
                    network.set_weights(original_weights)

                x, y = data.to(device), target.to(device)
                original_grad = network.get_grad(x, y, loss_func, zero_grad=False).detach().cpu().numpy()
                original_grad = original_grad / np.linalg.norm(original_weights)

                network.random_teleport(cob_range=x_axis[i])

                teleported_weights = network.get_weights().detach().cpu().numpy()
                teleported_grad = network.get_grad(x, y, loss_func, zero_grad=False).detach().cpu().numpy()
                teleported_grad = teleported_grad / np.linalg.norm(teleported_weights)

                diff = abs(np.linalg.norm(original_grad)-np.linalg.norm(teleported_grad))
                to_compute_mean.append(diff)

            # Early stop to prevent NaNs
            if (batch_idx+1) % limit_batches == 0:
                break

        variance.append(np.std(to_compute_mean))
        differences.append(np.mean(to_compute_mean))

    variance = np.array(variance)
    differences = np.array(differences)
    x_axis = np.array(x_axis)

    plt.errorbar(x_axis, differences, yerr=variance)
    plt.plot(x_axis, differences)

    plt.title(f'{network_descriptor}')
    plt.ylabel('| ||dW||/||W|| - ||d(TW)||/||TW|| |')
    plt.xlabel('CoB range')
    plt.show()


if __name__ == '__main__':
    from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
    from neuralteleportation.training import experiment_setup
    from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
    from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
    from neuralteleportation.models.model_zoo.vggcob import vgg16_bnCOB
    from neuralteleportation.models.model_zoo.resnetcob import resnet18COB

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_shape = (32, 3, 32, 32)
    trainset, valset, testset = experiment_setup.get_dataset_subsets("cifar10")
    train_loader = DataLoader(trainset, batch_size=input_shape[0], shuffle=True)

    # MLP
    pbar = tqdm(enumerate(train_loader))
    mlp = MLPCOB(input_shape=(3, 32, 32), num_classes=10).to(device=device)
    mlp = NeuralTeleportationModel(network=mlp, input_shape=input_shape)
    plot_difference_teleported_gradients(network=mlp, pbar=pbar, network_descriptor='MLP on CIFAR-10', device=device)
    pbar.close()

    # VGG
    pbar = tqdm(enumerate(train_loader))
    vgg = vgg16_bnCOB(num_classes=10).to(device=device)
    vgg = NeuralTeleportationModel(network=vgg, input_shape=input_shape)
    plot_difference_teleported_gradients(network=vgg, pbar=pbar, network_descriptor='VGG on CIFAR-10', device=device)
    pbar.close()

    # ResNet
    pbar = tqdm(enumerate(train_loader))
    resnet = resnet18COB(num_classes=10).to(device=device)
    resnet = NeuralTeleportationModel(network=resnet, input_shape=input_shape)
    plot_difference_teleported_gradients(network=resnet, pbar=pbar, network_descriptor='ResNet on CIFAR-10',
                                         device=device)
    pbar.close()

    # DenseNet
    pbar = tqdm(enumerate(train_loader))
    densenet = densenet121COB(num_classes=10).to(device=device)
    densenet = NeuralTeleportationModel(network=densenet, input_shape=input_shape)
    plot_difference_teleported_gradients(network=densenet, pbar=pbar, network_descriptor='DenseNet on CIFAR-10',
                                         device=device)
    pbar.close()
