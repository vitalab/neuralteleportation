import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

# ANSI escape code for colored console text
red = "\033[31m"
reset = "\033[0m"


def dot_product(network, dataset, nb_teleport=200, network_descriptor='',
                sampling_types=['usual', 'symmetric', 'negative', 'zero'],
                batch_sizes = [8, 16, 32, 64],
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
    tol = 1e-2
    loss_func = torch.nn.CrossEntropyLoss()

    aggregator = pd.DataFrame(columns=['sampling type', 'batch size',
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
            w1 = model.get_weights().detach().numpy()

            for power in range(-2, 2):
                cob = 10 ** power
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
                    w2 = model.get_weights().detach().numpy()

                    # get teleportation vector
                    micro_teleport_vec = (w2 - w1)

                    random_vector = torch.rand(grad.shape, dtype=torch.float)-0.5
                    random_vector2 = torch.rand(grad.shape, dtype=torch.float)-0.5

                    # Normalized scalar products
                    dot_prod = np.longfloat(np.dot(grad, micro_teleport_vec) /
                                            (np.linalg.norm(grad)*np.linalg.norm(micro_teleport_vec)))

                    angle = np.degrees(np.arccos(dot_prod))

                    rand_dot_prod = np.longfloat(np.dot(grad, random_vector) /
                                                 (np.linalg.norm(grad)*np.linalg.norm(random_vector)))
                    rand_angle = np.degrees(np.arccos(rand_dot_prod))

                    rand_rand_dot_prod = np.longfloat(np.dot(random_vector2, random_vector) /
                                                    (np.linalg.norm(random_vector2) *
                                                    np.linalg.norm(random_vector)))
                    rand_rand_angle = np.degrees(np.arccos(rand_rand_dot_prod))

                    rand_micro_dot_prod = np.longfloat(np.dot(random_vector2, micro_teleport_vec) /
                                                      (np.linalg.norm(random_vector2) *
                                                       np.linalg.norm(micro_teleport_vec)))
                    # to degress
                    rand_micro_angle = np.degrees(np.arccos(rand_micro_dot_prod))

                    failed = (not np.allclose(dot_prod, 0, atol=tol))
                    rand_failed = (not np.allclose(rand_dot_prod, 0, atol=tol))
                    target_angle = 90

                    angle_results.append(angle)
                    rand_angle_results.append(rand_angle)
                    rand_rand_angle_results.append(rand_rand_angle)
                    rand_micro_angle_results.append(rand_micro_angle)

                angle_results = np.array(angle_results)
                rand_angle_results = np.array(rand_angle_results)
                rand_rand_angle_results = np.array(rand_rand_angle_results)
                rand_micro_angle_results = np.array(rand_micro_angle_results)

                # Append resuslts to dataframe for further ploting
                aggregator = aggregator.append({'sampling type' : sampling_type,
                                               'batch size' : batch_size,
                                               'COB range' : cob,
                                               'Micro-teleportation vs Gradient' : angle_results.mean(),
                                               'Micro-teleportation vs Gradient std' : angle_results.std(),
                                               'Gradient vs Random Vector' : rand_angle_results.mean(),
                                               'Gradient vs Random Vector std' : rand_angle_results.std(),
                                               'Random Vector vs  Random Vector' : rand_rand_angle_results.mean(),
                                               'Random Vector vs  Random Vector std' : rand_rand_angle_results.std(),
                                               'Micro-teleportation vs Random Vector' : rand_micro_angle_results.mean(),
                                               'Micro-teleportation vs Random Vector std' : rand_micro_angle_results.std()},
                                              ignore_index=True)

                print(f'The result of the scalar product between the gradient and a micro-teleporation vector is: '
                      f'{red * failed}{np.round(angle_results.mean(), abs(int(np.log10(tol))))}',
                      f' (!=0 => FAILED!)' * failed,
                      f'{reset}',
                      f' using {sampling_type} sampling type',
                      f', the angle is {angle}°',
                      f', the delta in angle is {angle - target_angle}°\n',
                      f'The result of the scalar product  between the gradient and a random vector is: ',
                      f'{red * rand_failed}{rand_angle_results.mean()}',
                      f' (FAILED!)' * rand_failed,
                      f'{reset}',
                      f', and the angle is {rand_angle}°',
                      f', the delta in angle is {rand_angle - target_angle}°\n',
                      sep='')

                # if not np.isnan(np.sum(angle_results)):
                #     delta = np.maximum(1.0, rand_rand_angle_results.std() * 3)
                #     x_min = 90 - delta
                #     x_max = 90 + delta
                #
                #     plt.subplot(4, 1, 1)
                #
                #     bin_height, bin_boundary = np.histogram(np.array(angle_results))
                #     width = bin_boundary[1] - bin_boundary[0]
                #     bin_height = bin_height / float(max(bin_height))
                #     plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.05))
                #     plt.title(f'{network_descriptor}: Sampling type: {sampling_type}, cob range: {cob}\n'
                #               f'{iterations:} iter, batch size: {batch_size}')
                #     plt.legend(['Micro-teleportation\n vs \n Gradient'])
                #     plt.xlim(x_min, x_max)
                #
                #     bin_height, bin_boundary = np.histogram(np.array(rand_micro_angle_results))
                #     width = bin_boundary[1] - bin_boundary[0]
                #     bin_height = bin_height / float(max(bin_height))
                #     plt.subplot(4, 1, 2)
                #     plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
                #     plt.xlim(x_min, x_max)
                #     plt.legend(['Micro-teleportation\n vs \n Random Vector'])
                #
                #     bin_height, bin_boundary = np.histogram(np.array(rand_angle_results))
                #     width = bin_boundary[1] - bin_boundary[0]
                #     bin_height = bin_height / float(max(bin_height))
                #     plt.subplot(4, 1, 3)
                #     plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
                #     plt.xlim(x_min, x_max)
                #     plt.legend(['Gradient\n vs \n Random Vector'])
                #
                #     bin_height, bin_boundary = np.histogram(np.array(rand_rand_angle_results))
                #     width = bin_boundary[1] - bin_boundary[0]
                #     bin_height = bin_height / float(max(bin_height))
                #     plt.subplot(4, 1, 4)
                #     plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
                #     plt.xlim(x_min, x_max)
                #     plt.legend(['Random Vector\n vs \n Random Vector'])
                #
                #     plt.xlabel('Angle in degrees')
                #     plt.savefig(f'images/{network_descriptor}_Samp_type_{sampling_type}'
                #                 f'_cob_{cob}_iter_{iterations}_batch_size_{batch_size}.png'
                #                 )
                #     plt.show()
                #
                # else:
                #     print(f'{red}Something went wrong while generating the graphic!:{reset}',
                #           f'angle results contains NaN values' * np.isnan(np.sum(angle_results)),
                #           f'Teleported weights diverged to infinity' * (sum(np.isinf(w2)) > 0),
                #           f'{network_descriptor}: Sampling type: {sampling_type}, cob range: {cob}',
                #           f'{nb_teleport:} iter, batch size: {batch_size}\n',
                #           sep='\n')

    # Print bath stize series with error bars for each sampling type
    for sampling_type in sampling_types:
        for batch_size in batch_sizes:
            plt.errorbar(x=np.log10(aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                          (aggregator['batch size'] == batch_size)]['COB range']),
                         y=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                          (aggregator['batch size'] == batch_size)]['Micro-teleportation vs Gradient'],
                         yerr=aggregator.loc[(aggregator['sampling type'] == sampling_type) &
                                             (aggregator['batch size'] == batch_size)]['Micro-teleportation vs Gradient std']*3)

            plt.xlim((-2.2, 1.2))
            plt.xlabel('log(COB)')
            plt.ylabel('Theta')
            plt.title(f'{network_descriptor} - Sampling type: {sampling_type}, batch size: {batch_size}')
            plt.show()