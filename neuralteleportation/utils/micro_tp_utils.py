import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from tqdm import tqdm
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from torch import Tensor
from pathlib import Path

# ANSI escape code for colored console text
red = "\033[31m"
green = "\033[32m"
reset = "\033[0m"


def tensor_norm(t: Tensor) -> Tensor:
    """
    This method return the Euclidian norm (L2) of a tensor

    Args:
        t :                     the tensor for which we want the euclidian norm
    """
    return torch.sqrt(torch.sum(torch.pow(t, 2)))


def normalized_dot_product(t1: Tensor, t2: Tensor) -> Tensor:
    """
    This function returns the normalized scalar products between two tensors. In order to make the method
    device-agnostic, if the inputs are numpy arrays, they're converted to pytorch tensors

    Args:
        t1 :                     the tensor for which we want the euclidian norm
        t2 :                     the tensor for which we want the euclidian norm
    """
    if type(t2) is np.ndarray:
        t2 = torch.tensor(t2)
    if type(t1) is np.ndarray:
        t1 = torch.tensor(t1)
    return torch.matmul(t1, t2) / (tensor_norm(t1) * tensor_norm(t2))


def micro_teleportation_dot_product(network, dataset, nb_teleport=100, network_descriptor='',
                                    batch_sizes=[8, 16, 32, 64],
                                    criterion=None,
                                    device='cpu',
                                    verbose=False,
                                    random_data=False) -> None:
    """
    This method tests the scalar product between the teleporation line and the gradient, as well as between a random
    vector and the gradient for nullity. It then displays the histograms of the calculated scalar products. The
    method also aggregates all relevant micro teleportation data in a dataframe.

    Args:
        network :               the model which we wish to use to compute the micro-teleporations

        dataset:                 the dataset that will be used to calculate the gradient and get dimensions for the
                                neural teleportation model

        nb_teleport:            The number of time the network is teleported and the scalar product calculated. An
                                average is then calculated.

        network_descriptor:     String describing the content of the network

        batch_sizes:             Size of the minibatch used to perform gradient calculation

        criterion:              the loss function used to compute the gradient

        device:                 Device used to compute the network operations ('cpu' or 'cuda')

        verbose:                If true, the method will output extensive details about the calculated vectors and
                                aggregated data (mainly for debugging purposes)
    """

    # Arbitrary precision threshold for nullity comparison
    torch.set_printoptions(precision=10, sci_mode=True)
    tol = 1e-2
    cob = 0.001
    hist_dir = f'images/histograms/{network_descriptor}'
    marker_cover = Rectangle((0, 0), 0, 0, alpha=0.0)

    if torch.cuda.is_available():
        print(f'{green}Using CUDA{reset}')
        network = network.cuda()

    if (criterion is None):
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = criterion

    # Initialize the dataframe for data aggregation
    aggregator = pd.DataFrame(columns=['model name',
                                       'batch size',
                                       'COB range',
                                       'weights vector length',
                                       'Micro-teleportation vs Gradient',
                                       'Micro-teleportation vs Gradient std',
                                       'Gradient vs Random Vector',
                                       'Gradient vs Random Vector std',
                                       'Random Vector vs  Random Vector',
                                       'Random Vector vs  Random Vector std',
                                       'Micro-teleportation vs Random Vector',
                                       'Micro-teleportation vs Random Vector std'])

    for batch_size in batch_sizes:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        data, target = next(iter(dataloader))
        if random_data:
            data, target = torch.rand(data.shape), torch.rand(target.shape)

        # save the initial weights for further reset
        model = NeuralTeleportationModel(network=network, input_shape=data.shape)

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()

        if torch.cuda.is_available():
            w1 = model.get_weights().detach()
        else:
            w1 = model.get_weights().detach().numpy()

        angle_results = []
        rand_angle_results = []
        rand_rand_angle_results = []
        rand_micro_angle_results = []

        iterations = min(int(len(dataloader.dataset) / dataloader.batch_size), nb_teleport)

        for _ in tqdm(range(0, iterations)):

            # Get next data batch
            data, target = next(iter(dataloader))
            data, target = data.to(device), target.to(device)
            grad = model.get_grad(data, target, loss_func, zero_grad=False)

            # reset the weights for next teleportation
            model.set_weights(torch.tensor(w1))

            # teleport and get the new weights
            model = model.random_teleport(cob_range=cob)

            if torch.cuda.is_available():
                w2 = model.get_weights().detach()
            else:
                w2 = model.get_weights().detach().numpy()

            # get teleportation vector
            micro_teleport_vec = (w2 - w1)

            random_vector = torch.rand(grad.shape, dtype=torch.float) - 0.5
            random_vector2 = torch.rand(grad.shape, dtype=torch.float) - 0.5

            random_vector = random_vector.to(device)
            random_vector2 = random_vector2.to(device)

            # Normalized scalar products & angles calculations
            dot_prod = normalized_dot_product(grad, micro_teleport_vec)
            angle = np.degrees(torch.acos(dot_prod).cpu())

            rand_dot_prod = normalized_dot_product(grad, random_vector)
            rand_angle = np.degrees(torch.acos(rand_dot_prod).cpu())

            rand_rand_dot_prod = normalized_dot_product(random_vector2, random_vector)
            rand_rand_angle = np.degrees(torch.acos(rand_rand_dot_prod).cpu())

            rand_micro_dot_prod = normalized_dot_product(random_vector2, micro_teleport_vec)
            rand_micro_angle = np.degrees(torch.acos(rand_micro_dot_prod).cpu())

            # Perpendicularity assertion
            failed = (not torch.allclose(dot_prod, torch.tensor([0.0]).to(device), atol=tol))
            rand_failed = (not torch.allclose(rand_dot_prod, torch.tensor([0.0]).to(device), atol=tol))
            target_angle = 90.0

            angle_results.append(angle)
            rand_angle_results.append(rand_angle)
            rand_rand_angle_results.append(rand_rand_angle)
            rand_micro_angle_results.append(rand_micro_angle)

        angle_results = np.array(angle_results)
        rand_angle_results = np.array(rand_angle_results)
        rand_rand_angle_results = np.array(rand_rand_angle_results)
        rand_micro_angle_results = np.array(rand_micro_angle_results)

        # Append resuslts to dataframe for further ploting
        aggregator = aggregator.append({'model name' : network_descriptor,
                                        'batch size': batch_size,
                                        'COB range': cob,
                                        'weights vector length': len(w1),
                                        'Micro-teleportation vs Gradient': angle_results.mean(),
                                        'Micro-teleportation vs Gradient std': angle_results.std(),
                                        'Gradient vs Random Vector': rand_angle_results.mean(),
                                        'Gradient vs Random Vector std': rand_angle_results.std(),
                                        'Random Vector vs  Random Vector': rand_rand_angle_results.mean(),
                                        'Random Vector vs  Random Vector std': rand_rand_angle_results.std(),
                                        'Micro-teleportation vs Random Vector': rand_micro_angle_results.mean(),
                                        'Micro-teleportation vs Random Vector std': rand_micro_angle_results.std()},
                                       ignore_index=True)

        print(f'The angle between the gradient and a micro-teleporation vector is: '
              f'{red * failed}'
              f'{np.round(angle_results.mean(), abs(int(np.log10(tol))))}',
              f' (!=0 => FAILED!)' * failed,
              f'{reset}',
              f', the delta in angle is {angle - target_angle}°\n',
              f'The angle between the gradient and a random vector is: ',
              f'{red * rand_failed}{rand_angle_results.mean()}',
              f' (FAILED!)' * rand_failed,
              f'{reset}',
              f', the delta in angle is {rand_angle - target_angle}°\n',
              sep='')

        if verbose:
            print(aggregator.iloc[aggregator.last_valid_index()])
            if torch.cuda.is_available():
                print(f'w1: {w1}', f'nans: {torch.sum(torch.isnan(w1))}',
                      f'max: {torch.max(w1)}',
                      f'min: {torch.min(w1)}',
                      sep='\n')
                print(f'w2: {w2}',
                      f' nans: {torch.sum(torch.isnan(w2))}',
                      f'max: {torch.max(w2)}',
                      f'min: {torch.min(w2)}',
                      sep='\n')
            else:
                print(f'w1: {w1}', f'nans: {np.sum(np.isnan(w1))}',
                f'max: {np.max(w1)}',
                f'min: {np.min(w1)}',
                sep='\n')
                print(f'w2: {w2}',
                f' nans: {np.sum(np.isnan(w2))}',
                f'max: {np.max(w2)}',
                f'min: {np.min(w2)}',
                sep='\n')

        if not np.isnan(aggregator.loc[aggregator.last_valid_index(), 'Micro-teleportation vs Gradient']):
            delta = 0.25
            x_min = 90 - delta
            x_max = 90 + delta
            figsize = (10.0, 10.0)

            fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=figsize)
            fig.suptitle(f'{hist_dir}/{network_descriptor}_cob_{cob}_iter_{iterations}' +
                         f'_random_data' * random_data + f'_batch_size_{batch_size}')

            bin_height, bin_boundary = np.histogram(np.array(angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            ax0.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.01),
                    label='Micro-teleportation\n vs \n Gradient', handles=marker_cover)
            ax0.legend(handles=[marker_cover], loc='upper right', shadow=False, frameon=False)
            ax0.set_xlim(x_min, x_max)

            bin_height, bin_boundary = np.histogram(np.array(rand_micro_angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            ax1.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g',
                    label='Micro-teleportation\n vs \n Random Vector', handles=marker_cover)
            ax1.legend(loc='upper right', shadow=False, frameon=False, handlelength=0.0)
            ax1.set_xlim(x_min, x_max)

            bin_height, bin_boundary = np.histogram(np.array(rand_angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            ax2.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g',
                    label='Gradient\n vs \n Random Vector', handles=marker_cover)
            ax2.legend(loc='upper right', shadow=False, frameon=False)
            ax2.set_xlim(x_min, x_max)

            bin_height, bin_boundary = np.histogram(np.array(rand_rand_angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            ax3.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g',
                    label='Random Vector\n vs \n Random Vector', handles=marker_cover)
            ax3.legend(loc='upper right', shadow=False, frameon=False)
            ax3.set_xlim(x_min, x_max)

            plt.xlabel('Angle in degrees')

            Path(hist_dir).mkdir(parents=True, exist_ok=True)
            fig_name = f'{hist_dir}/{network_descriptor}_cob_{cob}_iter_{iterations}' + \
                       f'_random_data' * random_data + f'_batch_size_{batch_size}.pdf'
            plt.savefig(fig_name)
            plt.show()
        else:
            print(red)
            print(aggregator.iloc[aggregator.last_valid_index()])
            print(reset)
