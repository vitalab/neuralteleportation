import torch
import matplotlib.pyplot as plt
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
    return torch.sqrt(torch.sum(torch.pow(t, 2)))


def normalized_dot_product(t1: Tensor, t2: Tensor) -> Tensor:
    if type(t2) is np.ndarray:
        t2 = torch.tensor(t2)
    if type(t1) is np.ndarray:
        t1 = torch.tensor(t1)
    return torch.matmul(t1, t2) / (tensor_norm(t1) * tensor_norm(t2))


def dot_product(network, dataset, nb_teleport=200, network_descriptor='',
                sampling_types=['usual', 'symmetric', 'negative', 'zero'],
                batch_sizes=[8, 16, 32, 64],
                criterion=None,
                device='cpu') -> None:
    """
    This method tests the scalar product between the teleporation line and the gradient, as well as between a random
    vector and the gradient for nullity. It then displays the histograms of the calculated scalar products.

    Args:
        network :               the model to which we wish to assign weights

        input_shape :           the shape of the input.  By default, simulate batched of 100 grayscale 28x28 images
                                (it will be used by the networkgrapher of the model,
                                the values is not important for the test at hand)

        nb_teleport:            The number of time the network is teleported and the scalar product calculated. An
                                average is then calculated.

        network_descriptor:     String describing the content of the network

        sampling_types :        Teleportation sampling types

        device:                 Device used to compute the netork operations (Typically 'cpu' or 'cuda')
    """

    # Arbitrary precision threshold for nullity comparison
    torch.set_printoptions(precision=10, sci_mode=True)
    tol = 1e-2
    cobs = [0.001, 0.01, 0.1, 10]
    hist_dir = f'images/histograms/{network_descriptor}'
    series_dir = f'images/series/{network_descriptor}'

    if torch.cuda.is_available():
        print(f'{green}Using CUDA{reset}')
        network = network.cuda()

    if criterion is None:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = criterion

    aggregator = pd.DataFrame(columns=['sampling type',
                                       'batch size',
                                       'COB range',
                                       'Micro-teleportation vs Gradient',
                                       'Micro-teleportation vs Gradient std',
                                       'Gradient vs Random Vector',
                                       'Gradient vs Random Vector std',
                                       'Random Vector vs  Random Vector',
                                       'Random Vector vs  Random Vector std',
                                       'Micro-teleportation vs Random Vector',
                                       'Micro-teleportation vs Random Vector std'])

    for sampling_type in sampling_types:
        for batch_size in batch_sizes:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            data, target = next(iter(dataloader))

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

            for cob in cobs:
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
                    model.set_weights(w1)

                    # teleport and get the new weights
                    model.random_teleport(cob_range=cob, sampling_type=sampling_type)

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
                aggregator = aggregator.append({'sampling type': sampling_type,
                                                'batch size': batch_size,
                                                'COB range': cob,
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
                      f' using {sampling_type} sampling type',
                      f', the delta in angle is {angle - target_angle}°\n',
                      f'The angle between the gradient and a random vector is: ',
                      f'{red * rand_failed}{rand_angle_results.mean()}',
                      f' (FAILED!)' * rand_failed,
                      f'{reset}',
                      f', the delta in angle is {rand_angle - target_angle}°\n',
                      sep='')

                delta = np.maximum(1.0, rand_rand_angle_results.std() * 3)
                x_min = 90 - delta
                x_max = 90 + delta
                figsize = (10.0, 10.0)

                fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=figsize)
                fig.suptitle(f'{network_descriptor}: Sampling type: {sampling_type}, cob range: {cob}\n'
                             f'{iterations:} iter, batch size: {batch_size}')

                bin_height, bin_boundary = np.histogram(np.array(angle_results))
                width = bin_boundary[1] - bin_boundary[0]
                bin_height = bin_height / float(max(bin_height))
                ax0.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.05))
                ax0.legend(['Micro-teleportation\n vs \n Gradient'])
                ax0.set_xlim(x_min, x_max)

                bin_height, bin_boundary = np.histogram(np.array(rand_micro_angle_results))
                width = bin_boundary[1] - bin_boundary[0]
                bin_height = bin_height / float(max(bin_height))
                ax1.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
                ax1.set_xlim(x_min, x_max)
                ax1.legend(['Micro-teleportation\n vs \n Random Vector'])

                bin_height, bin_boundary = np.histogram(np.array(rand_angle_results))
                width = bin_boundary[1] - bin_boundary[0]
                bin_height = bin_height / float(max(bin_height))
                ax2.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
                ax2.set_xlim(x_min, x_max)
                ax2.legend(['Gradient\n vs \n Random Vector'])

                bin_height, bin_boundary = np.histogram(np.array(rand_rand_angle_results))
                width = bin_boundary[1] - bin_boundary[0]
                bin_height = bin_height / float(max(bin_height))
                ax3.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
                ax3.set_xlim(x_min, x_max)
                ax3.legend(['Random Vector\n vs \n Random Vector'])

                plt.xlabel('Angle in degrees')

                Path(hist_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{hist_dir}/{network_descriptor}_Samp_type_{sampling_type}'
                            f'_cob_{cob}_iter_{iterations}_batch_size_{batch_size}.png')
                plt.show()

    for sampling_type in sampling_types:
        for cob in cobs:
            plt.scatter(x=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                          (aggregator['COB range'] == cob)]['batch size'],
                         y=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                          (aggregator['COB range'] == cob)][
                             'Micro-teleportation vs Gradient'],
                        c='red',
                        marker='o')
            plt.errorbar(x=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                          (aggregator['COB range'] == cob)]['batch size'],
                         y=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                          (aggregator['COB range'] == cob)][
                             'Micro-teleportation vs Gradient'],
                         yerr=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                             (aggregator['COB range'] == cob)][
                                  'Micro-teleportation vs Gradient std'] * 3,
                         linestyle='None')

            plt.xlabel('Batch size')
            plt.ylabel('Theta')
            plt.title(f'{network_descriptor} - Sampling type: {sampling_type}, C.O.B.: {cob}')

            Path(series_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{series_dir}/{network_descriptor}_Samp_type_{sampling_type}'
                        f'_cob_{cob}.png')
            plt.show()



def dot_product_between_telportation(network, dataset,
                                     network_descriptor=None,
                                     criterion=None,
                                     normalized=False,
                                     reset_weights=False,
                                     device='cpu') -> None:

    series_dir = f'images/series_dot_prod_vs_cob/{network_descriptor}'

    if torch.cuda.is_available():
        print(f'{green}Using CUDA{reset}')
        network = network.cuda()

    if network_descriptor is None:
        network_descriptor = network.__name__

    # Prepare the range of COB to test
    cobs = np.linspace(1, 9, 10)
    # We don't want a COB of 1 since it produces no teleportation
    cobs[0] += 0.1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    data, target = next(iter(dataloader))
    model = NeuralTeleportationModel(network=network, input_shape=data.shape)

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    if torch.cuda.is_available():
        w1 = model.get_weights().detach()
    else:
        w1 = model.get_weights().detach().numpy()

    dot_product_results = []
    for cob in cobs:
        # reset the weights for next teleportation
        if reset_weights:
            model.set_weights(w1)
        else:
            if torch.cuda.is_available():
                w1 = model.get_weights().detach()
            else:
                w1 = model.get_weights().detach().numpy()

        # teleport and get the new weights
        model.random_teleport(cob_range=cob, sampling_type='usual')
        if torch.cuda.is_available():
            w2 = model.get_weights().detach()
        else:
            w2 = model.get_weights().detach().numpy()

        if normalized:
            dot_product_results.append(normalized_dot_product(w1, w2))
        else:
            dot_product_results.append(torch.matmul(
                torch.tensor(w1).to(device), torch.tensor(w2).to(device)))

    # plt.scatter(np.log10(cobs), dot_product_results, c='red', marker='o')
    plt.plot(np.log10(cobs), dot_product_results)
    plt.title(f'Sacalar product between original and \nteleported weights with '
              f'respect to COB\'s order of magnitude\n{network_descriptor}, '
              f'reset weights: {reset_weights}')

    plt.ylabel('Sacalar product')
    plt.xlabel('log10(COB)')

    Path(series_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{series_dir}/{network_descriptor}_Samp_type_usual.png')
    plt.show()
