import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from models.model_zoo import vggcob, resnetcob
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

# ANSI escape code for colored console text
red = "\033[31m"
reset = "\033[0m"


def dot_product(network, dataset, nb_teleport=200, network_descriptor='',
                sampling_types=['usual', 'symmetric', 'negative', 'zero'], device='cpu') -> None:
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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    data, target = next(iter(dataloader))

    model = NeuralTeleportationModel(network=network, input_shape=data.shape)

    w1 = model.get_weights().detach().numpy()

    iterations = range(0, nb_teleport)
    loss_func = torch.nn.CrossEntropyLoss()

    # Arbitrary precision threshold for nullity comparison
    tol = 1e-2

    for sampling_type in sampling_types:
        for power in range(-2, 2):
            cob = 10 ** power
            angle_results = []
            rand_angle_results = []
            rand_rand_angle_results = []
            rand_micro_angle_results = []
            for _ in iterations:
                data, target = data.to(device), target.to(device)
                grad = model.get_grad(data, target, loss_func, zero_grad=False)

                model.set_weights(w1)
                model.random_teleport(cob_range=cob, sampling_type=sampling_type)
                w2 = model.get_weights().detach().numpy()
                micro_teleport_vec = (w2 - w1)

                random_vector = torch.rand(grad.shape, dtype=torch.float)-0.5
                random_vector2 = torch.rand(grad.shape, dtype=torch.float)-0.5

                # Normalized scalar product
                dot_prod = np.longdouble(np.dot(grad, micro_teleport_vec) /
                                        (np.linalg.norm(grad)*np.linalg.norm(micro_teleport_vec)))
                angle = np.degrees(np.arccos(dot_prod))

                rand_dot_prod = np.longdouble(np.dot(grad, random_vector) /
                                             (np.linalg.norm(grad)*np.linalg.norm(random_vector)))
                rand_angle = np.degrees(np.arccos(rand_dot_prod))

                rand_rand_dot_prod = np.longdouble(np.dot(random_vector2, random_vector) /
                                                  (np.linalg.norm(random_vector2)*np.linalg.norm(random_vector)))
                rand_rand_angle = np.degrees(np.arccos(rand_rand_dot_prod))

                rand_micro_dot_prod = np.longdouble(np.dot(random_vector2, micro_teleport_vec) /
                                                  (np.linalg.norm(random_vector2)*np.linalg.norm(micro_teleport_vec)))
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

            print(f'The result of the scalar product between the gradient and a micro-teleporation vector is: '
                  f'{red * failed}{np.round(angle_results.mean(), abs(int(np.log10(tol))))}',
                  f' (!=0 => FAILED!)' * failed,
                  f'{reset}',
                  f' using {sampling_type} sampling type',
                  f', the angle is {angle}°',
                  f', the delta in angle is {angle - target_angle}°\n',
                  f'The result of the scalar product  between the gradient and a random vector is: ',
                  f'{red * rand_failed}{rand_angle_results.mean()}',
                  f' (!=0 => FAILED!)' * rand_failed,
                  f'{reset}',
                  f', and the angle is {rand_angle}°',
                  f', the delta in angle is {rand_angle - target_angle}°\n',
                  sep='')

            # This conditional display is necessary because some sampling type/COB combinations produce such a narrow
            # distribution for micro-teleportation that pyplot is not able to display them at all
            delta = np.maximum(1.0, rand_rand_angle_results.std()*3)
            x_min = 90-delta
            x_max = 90+delta

            plt.subplot(4, 1, 1)

            bin_height, bin_boundary = np.histogram(np.array(angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.05))
            plt.title(f'{network_descriptor}: Sampling type: {sampling_type}, cob range: {cob}, 'f'{nb_teleport:} iter')
            plt.legend(['Micro-teleportation\n vs \n Gradient'])
            plt.xlim(x_min, x_max)

            bin_height, bin_boundary = np.histogram(np.array(rand_micro_angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            plt.subplot(4, 1, 2)
            plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
            plt.xlim(x_min, x_max)
            plt.legend(['Micro-teleportation\n vs \n Random Vector'])

            bin_height, bin_boundary = np.histogram(np.array(rand_angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            plt.subplot(4, 1, 3)
            plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
            plt.xlim(x_min, x_max)
            plt.legend(['Gradient\n vs \n Random Vector'])

            bin_height, bin_boundary = np.histogram(np.array(rand_rand_angle_results))
            width = bin_boundary[1] - bin_boundary[0]
            bin_height = bin_height / float(max(bin_height))
            plt.subplot(4, 1, 4)
            plt.bar(bin_boundary[:-1], bin_height, width=np.maximum(width, 0.1), color='g')
            plt.xlim(x_min, x_max)
            plt.legend(['Random Vector\n vs \n Random Vector'])

            plt.xlabel('Angle in degrees')
            plt.show()

def train(model, criterion, train_dataset, val_dataset=None, optimizer=None, metrics=None, epochs=10, batch_size=32,
          device='cpu'):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        train_step(model, criterion, optimizer, train_loader, epoch, device=device)
        if val_dataset:
            val_res = test(model, criterion, metrics, val_dataset, device=device)
            print("Validation: {}".format(val_res))


def train_step(model, criterion, optimizer, train_loader, epoch, device='cpu'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       batch_idx / len(train_loader), loss.item()))


def test(model, criterion, metrics, dataset, batch_size=32, device='cpu'):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            results['loss'].append(criterion(output, target).item())

            if metrics is not None:
                batch_results = compute_metrics(metrics, y=target, y_hat=output, to_tensor=False)
                for k in batch_results.keys():
                    results[k].append(batch_results[k])

    results = pd.DataFrame(results)
    return dict(results.mean())


def compute_metrics(metrics, y_hat, y, prefix='', to_tensor=True):
    results = {}
    for metric in metrics:
        m = metric(y_hat, y)
        if to_tensor:
            m = torch.tensor(m)
        results[prefix + metric.__name__] = m
    return results


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    import torch.nn as nn

    trans = list()
    trans.append(transforms.Resize(size=224))
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    mnist_train = MNIST('/tmp', train=True, download=True, transform=trans)
    mnist_val = MNIST('/tmp', train=False, download=True, transform=trans)
    mnist_test = MNIST('/tmp', train=False, download=True, transform=trans)

    # GGS: reducing the dataset size since cuda is not available on local system due to unidentified bug as of
    # may 28th 2020, this should be taken down once CUDA problem has been addressed
    mnist_train.data = mnist_train.data[:50, :, :]
    mnist_val.data = mnist_val.data[:50, :, :]
    mnist_test.data = mnist_test.data[:50, :, :]

    vgg_model = vggcob.vgg11COB(input_channels=1)

    optim = torch.optim.Adam(params=vgg_model.parameters(), lr=.01)
    metrics = [accuracy]
    loss = nn.CrossEntropyLoss()
    train(vgg_model, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_val, optimizer=optim, metrics=metrics,
          epochs=1, device='cpu', batch_size=16)
    print(test(vgg_model, loss, metrics, mnist_test))

    dot_product(vgg_model, dataset=mnist_test, network_descriptor='VGG')
