# This script validates that the scalar product between a line defined by a set of weights and its teleporation
# and the gradient calculated a that point is null. It does this for many sampling types
# (see change get_random_cob in changeofbaseutils.py)

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

import numpy as np
import torch
import matplotlib.pyplot as plt

red = "\033[31m"
reset = "\033[0m"


def test_dot_product(network, input_shape=(1, 1, 28, 28), teleport_iterations=200) -> None:
    """
    This method tests the scalar produt between the teleporation line and the gradient, as well as between a random
    vector and the gradient for nullity. It then displays the histograms of the calculated scalar products.

    Args:
        network :               the model to which we wish to assigne weights

        input_shape :           the shape of the input
                                (it will be used by the networkgrapher of the model,
                                the values is not important for the test at hand)

        teleport_iterations:    The number of time the network is teleported and the scalar product calculated. An
                                average is then calculated.
    """

    model = NeuralTeleportationModel(network=network, input_shape=input_shape)
    x = torch.rand(input_shape, dtype=torch.float)
    y = torch.rand((1, 1, 10, 10), dtype=torch.float)
    w1 = model.get_weights().detach().numpy()
    sampling_types = ['usual', 'symmetric', 'negative', 'zero']

    iterations = range (0, teleport_iterations)
    loss_func = torch.nn.MSELoss()

    grad = model.get_grad(x, y, loss_func, zero_grad=False)

    for sampling_type in sampling_types:
        for pow in range(-5, 2):
            cob = 10 ** pow
            dot_prod_results = []
            rand_dot_prod_results = []
            for _ in iterations:
                model.set_weights(w1)
                model.random_teleport(cob_range=cob, sampling_type=sampling_type)
                w2 = model.get_weights().detach().numpy()

                # Normalized scalar product
                dot_prod = np.longfloat(np.dot(grad, (w2 - w1))/(np.linalg.norm(grad)*np.linalg.norm((w2 - w1))))
                angle = np.degrees(np.arccos(dot_prod))

                # Create a random vector to compare repectives scalar products
                random_vector = torch.rand(grad.shape, dtype=torch.float)
                rand_dot_prod = np.longfloat(np.dot(grad, random_vector)/
                                             (np.linalg.norm(grad)*np.linalg.norm(random_vector)))
                rand_angle = np.degrees(np.arccos(rand_dot_prod))

                # Arbitrary precision threshold for nullity comparison
                tol = 1e-2
                failed = (not np.allclose(dot_prod, 0, atol=tol))
                rand_failed = (not np.allclose(rand_dot_prod, 0, atol=tol))
                target_angle = 90

                dot_prod_results.append(dot_prod)
                rand_dot_prod_results.append(rand_dot_prod)

            dot_prod_results = np.array(dot_prod_results)
            rand_dot_prod_results = np.array(rand_dot_prod_results)

            print(f'The result of the scalar product between the gradient and a micro-teleporation vector is: '
                  f'{red * failed}{np.round(dot_prod_results.mean(), abs(int(np.log10(tol))))}',
                  f' (!=0 => FAILED!)' * failed,
                  f'{reset}',
                  f' using {sampling_type} sampling type',
                  f', the angle is {angle}°',
                  f', the delta in angle is {angle - target_angle}°\n',
                  f'The scalar product  between the gradient and a random vector is: ',
                  f'{red * rand_failed}{rand_dot_prod_results.mean()}',
                  f' (!=0 => FAILED!)' * rand_failed,
                  f'{reset}',
                  f', and the angle is {rand_angle}°',
                  f', the delta in angle is {rand_angle - target_angle}°\n',
                  sep='')

            bins = 'fd'
            # This conditionnal display is necessary because some sampling type/COB combinations produce such a narrow
            # distribution for micro-teleportation that pyplot is not able to display them at all
            if np.log10(rand_dot_prod_results.var() / dot_prod_results.var()) < 3.7:
                plt.hist(dot_prod_results, color='b', bins=bins)
                plt.hist(rand_dot_prod_results, color='g', bins=bins)
                plt.legend(['Micro-teleportation', 'Random vector'])
                plt.title(f'Sampling type: {sampling_type}, cob range: {cob}\n'
                          f'{teleport_iterations} iterations')
                plt.text(dot_prod_results.mean(), plt.ylim()[1]/2,
                         f'Micro: $\\mu={dot_prod_results.mean():.3e}$,\n'
                         f'$\\sigma={dot_prod_results.var():.3e}$')
                plt.text(rand_dot_prod_results.mean(), plt.ylim()[1]/2,
                         f'Random: $\\mu={rand_dot_prod_results.mean():.3e}$,\n'
                         f'$\\sigma={rand_dot_prod_results.var():.3e}$')
                plt.show()
            else:
                # Mirco-teleportation
                plt.hist(np.array(dot_prod_results), color='b', bins=bins)
                plt.title(f'Micro-teleportation - Sampling type: {sampling_type}, cob range: {cob}\n'
                          f'{teleport_iterations} iterations')
                plt.text(dot_prod_results.mean(), plt.ylim()[1]/2,
                         f'Micro: $\\mu={dot_prod_results.mean():.3e}$,\n'
                         f'$\\sigma={dot_prod_results.var():.3e}$')
                plt.show()

                # Random vector
                plt.hist(np.array(rand_dot_prod_results), color='g', bins=bins)
                plt.title(f'Random vector - Sampling type: {sampling_type}, cob range: {cob}\n'
                          f'{teleport_iterations} iterations')
                plt.text(rand_dot_prod_results.mean(), plt.ylim()[1]/2,
                         f'Random: $\\mu={rand_dot_prod_results.mean():.3e}$,\n'
                         f'$\\sigma={rand_dot_prod_results.var():.3e}$')
                plt.show()

if __name__ == '__main__':
    import torch.nn as nn
    from torch.nn.modules import Flatten
    from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules
    plt.close('all')

    cnn_model = torch.nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    mlp_model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    cnn_model = swap_model_modules_for_COB_modules(cnn_model)
    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    test_dot_product(network=mlp_model)
    test_dot_product(network=cnn_model)
