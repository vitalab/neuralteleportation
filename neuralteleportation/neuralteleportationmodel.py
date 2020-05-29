from typing import Tuple, Callable, Union, List

import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.merge import Add, Concat
from neuralteleportation.layers.neuron import NeuronLayerMixin
from neuralteleportation.changeofbasisutils import get_random_cob
from neuralteleportation.network_graph import NetworkGrapher


class NeuralTeleportationModel(nn.Module):
    """
        NeuralTelportationModel allows the teleportation of a nn.Module.

    Args:
        network (nn.Module):  Network to be wrapped for teleportation.
        input_shape (tuple): input shape used to compute the network graph.
    """

    def __init__(self, network: nn.Module, input_shape: Tuple) -> None:
        super(NeuralTeleportationModel, self).__init__()
        self.network = network

        device = next(self.network.parameters()).device
        sample_input = torch.rand(input_shape).to(device)

        self.grapher = NetworkGrapher(network, sample_input)
        self.graph = self.grapher.get_graph()

    def forward(self, x):
        return self.network(x)

    def initialize_cob(self):
        size = 0
        neuron_layers = self.get_neuron_layers()
        for i in range(len(neuron_layers) - 1):
            size += neuron_layers[i].out_features

        cob = torch.ones(size)

        self.set_change_of_basis(cob)
        self.apply_cob()

    def generate_random_cob(self, cob_range=10):
        size = 0
        neuron_layers = self.get_neuron_layers()
        for i in range(len(neuron_layers) - 1):
            size += neuron_layers[i].out_features

        return get_random_cob(range=cob_range, size=size, requires_grad=True)

    def get_random_change_of_basis(self, basis_range=10):
        """
          Compute random change of basis for every layer in the network.
        """

        current_cob = None  # Cob for last neuron layer to be applied following to non-neuron layers ie. Activations

        for i, layer in enumerate(self.graph):
            if isinstance(layer['module'], NeuronLayerMixin):
                # Check if this is the last neuron layer
                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    current_cob = layer['module'].get_output_cob()

                # Check if this is the first neuron layer
                elif not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[:i]]).any():
                    initial_cob = layer['module'].get_input_cob()
                    layer['prev_cob'] = initial_cob

                    # Apply change of basis for all previous layers.
                    for l in self.graph[:i]:
                        l['prev_cob'] = initial_cob
                        l['cob'] = initial_cob
                    current_cob = layer['module'].get_cob(basis_range=basis_range)
                else:
                    current_cob = layer['module'].get_cob(basis_range=basis_range)

            if isinstance(layer['module'], Add):
                connection_layer_index = min(layer['in'])  # Get the first layer

                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    '''If there is no layer after, previous layer must be ones. '''
                    raise ValueError("NOT SUPPORTED YET: Must have neuron layer after residual connection")

                current_cob = self.graph[connection_layer_index]['cob']

            if isinstance(layer['module'], Concat):
                '''If the layer is concatenation the change of basis for this layer is the concatenation of all change
                   of basis of previous connected layers.
                '''
                previous_layer_indexes = self.graph[i]['in']
                current_cob = np.concatenate([self.graph[j]['cob'] for j in previous_layer_indexes])

            if i > 0:
                # if multiple inputs, get input that is i-1 else get the first input
                input_layer_index = max(layer['in'])
                layer['prev_cob'] = self.graph[input_layer_index]['cob']

            layer['cob'] = current_cob

    def random_teleport(self, cob_range=10):
        """
          Applies random change of basis to each of the network layers.
        """
        self.get_random_change_of_basis(cob_range)
        self.teleport()

    def teleport(self):
        for k, layer in enumerate(self.graph):
            layer['module'].apply_cob(prev_cob=layer['prev_cob'], next_cob=layer['cob'])

    def apply_cob(self):
        """
        Apply change of basis for layers without teleporting.
        Only changes the chages of basis for activation layer, pooling, ...

        """
        for k, layer in enumerate(self.graph):
            if not isinstance(layer['module'], NeuronLayerMixin):
                layer['module'].apply_cob(prev_cob=layer['prev_cob'], next_cob=layer['cob'])

    def set_change_of_basis(self, cob, contains_ones=False):
        """
            Set the change of basis for the network.
        """

        counter = 0
        current_cob = None  # Cob for last neuron layer to be applied following to non-neuron layers ie. Activations

        for i, layer in enumerate(self.graph):
            if isinstance(layer['module'], NeuronLayerMixin):
                # Check if this is the last neuron layer
                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    if contains_ones:
                        # TODO check that these are ones.
                        current_cob = cob[counter: counter + layer['module'].out_features]
                        counter += layer['module'].out_features
                    else:
                        current_cob = layer['module'].get_output_cob()

                # Check if this is the first neuron layer
                elif not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[:i]]).any():
                    if contains_ones:
                        # TODO check that these are ones.
                        initial_cob = cob[counter: counter + layer['module'].in_features]
                        counter += layer['module'].in_features
                    else:
                        initial_cob = layer['module'].get_input_cob()
                    layer['prev_cob'] = initial_cob

                    # Apply change of basis for all previous layers.
                    for l in self.graph[:i]:
                        l['prev_cob'] = initial_cob
                        l['cob'] = initial_cob

                    current_cob = cob[counter: counter + layer['module'].out_features]
                    counter += layer['module'].out_features
                else:
                    current_cob = cob[counter: counter + layer['module'].out_features]
                    counter += layer['module'].out_features

            if isinstance(layer['module'], Add):
                connection_layer_index = min(layer['in'])  # Get the first layer

                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    '''If there is no layer after, previous layer must be ones. '''
                    raise ValueError("NOT SUPPORTED YET: Must have neuron layer after residual connection")

                current_cob = self.graph[connection_layer_index]['cob']

            if isinstance(layer['module'], Concat):
                '''If the layer is concatenation the change of basis for this layer is the concatenation of all change
                   of basis of previous connected layers.
                '''
                previous_layer_indexes = self.graph[i]['in']
                current_cob = np.concatenate([self.graph[j]['cob'] for j in previous_layer_indexes])

            if i > 0:
                # if multiple inputs, get input that is i-1 else get the first input
                input_layer_index = max(layer['in'])
                layer['prev_cob'] = self.graph[input_layer_index]['cob']

            layer['cob'] = current_cob

    def reset_weights(self):
        """Reset all layers."""

        def reset(m):
            for m in self.modules():
                getattr(m, 'reset_parameters', lambda: None)()

        self.apply(reset)

    def get_neuron_layers(self) -> List:
        """
            Get the layer in the network that contain weights (therefore change of basis)

        Returns:
            List of nn.Modules
        """
        return [l for l in self.grapher.ordered_layers if isinstance(l, NeuronLayerMixin)]

    def get_weights(self, concat: bool = True, flatten=True, bias=True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
            Return model weights

        Args:
            concat (bool): if true weights are returned as concatenated torch tensor,
                            else in the form of a list of Tensors

        Returns:
            torch.Tensor or list containing model weights

        """
        w = []

        for k, layer in enumerate(self.get_neuron_layers()):
            w.extend(layer.get_weights(flatten=flatten, bias=bias))

        if concat:
            return torch.cat(w)
        else:
            return w

    def set_weights(self, weights: Union[torch.Tensor, np.ndarray]):
        """
            Set weights to the network.

        Args:
            weights (Union[torch.Tensor, np.ndarray]): weights to set.
        """
        counter = 0
        for k, layer in enumerate(self.get_neuron_layers()):
            nb_params = layer.get_nb_params()
            w = weights[counter:counter + nb_params]
            layer.set_weights(w)
            counter += nb_params

    def get_grad(self, data: torch.Tensor, target: torch.Tensor, loss_fn: Callable,
                 concat: bool = True, zero_grad: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
            Return model gradients for data, target and loss function.
        Args:
            data (torch.Tensor): input data for the network
            target (torch.Tensor): target ouput
            loss_fn (Callable):
            concat (bool): if true weights are returned as concatenated torch tensor,
                            else in the form of a list of Tensors
            zero_grad (bool): if true gradients are reset before computing them on new data.

        Returns:
            torch.Tensor or list containing model weights

        """
        if zero_grad:
            self.network.zero_grad()

        grad = []
        output = self.network(data)
        loss = loss_fn(output, target)
        loss.backward()
        for k, layer in enumerate(self.get_neuron_layers()):
            grad.append(layer.weight.grad.flatten())

            if layer.bias is not None:
                grad.append(layer.bias.grad.flatten())

        if concat:
            return torch.cat(grad)
        else:
            return grad

    def get_cob(self, concat=True, contain_ones=False):
        """
            Get change of basis for network.

        Returns:
            List of change of basis for each layer.
        """
        cob = []
        for i, layer in enumerate(self.graph):
            if isinstance(layer['module'], NeuronLayerMixin):
                # Last layer
                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    if contain_ones:
                        cob.append(layer['cob'])
                # First layer
                elif not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[:i]]).any():
                    if contain_ones:
                        cob.append(layer['prev_cob'])
                    cob.append(layer['cob'])
                else:
                    cob.append(layer['cob'])

        if concat:
            return torch.cat(cob)
        else:
            return cob

    def calculate_cob(self, initial_weights, target_weights, concat=True):
        layers = self.get_neuron_layers()

        cob = [layers[0].get_input_cob()]
        for i in range(len(layers) - 2):
            t = layers[i].calculate_cob(initial_weights[i], target_weights[i], cob[-1])
            cob.append(t)

        print("Last layer")
        print(layers[-2].out_features)
        t = []
        for i in range(layers[-2].out_features):
            w0 = initial_weights[-2][i, :]
            w0_hat = target_weights[-2][i, :]
            w1 = initial_weights[-1][:, i]
            w1_hat = target_weights[-1][:, 1]
            t0 = cob[-1]
            t2 = layers[-1].get_output_cob()

            print(w0.shape)
            print(w1.shape)
            print(w0_hat.shape)
            print(w1_hat.shape)
            print(t0.shape)
            print(t2.shape)

            ti = torch.tensor(1.1)

            eta = 0.001

            for _ in range(20):
                print((w0 / t0).dot(w0 / t0))
                print((w0 / t0).dot(w0_hat))
                print((w1 * t2).dot(w1 * t2))
                print((w1 * t2).dot(w1_hat))
                grad = (2 * ti * (w0 / t0).dot(w0 / t0) -
                        2 * (w0 / t0).dot(w0_hat) -
                        2 * torch.pow(ti, -3) * (w1 * t2).dot(w1 * t2) +
                        2 * torch.pow(ti, -2) * (w1 * t2).dot(w1_hat))
                print("ti: ", ti)
                print("grad: ", grad)
                ti = ti - eta * grad

            print("final ti: ", ti)
            t.append(ti)

        cob.append(torch.tensor(t))

        cob.pop(0)
        # cob.pop(-1)

        if concat:
            return torch.cat(cob)
        else:
            return cob


if __name__ == '__main__':
    from neuralteleportation.models.generic_models.test_models import *
    from neuralteleportation.models.generic_models.residual_models import *
    from neuralteleportation.models.generic_models.dense_models import *
    from neuralteleportation.models.model_zoo.resnet import *

    model = resnet18(pretrained=False)
    sample_input_shape = (1, 3, 224, 224)

    model = NeuralTeleportationModel(network=model, input_shape=sample_input_shape)

    sample_input = torch.rand(sample_input_shape)

    pred1 = model(sample_input)
    w1 = model.get_weights()
    model.random_teleport(cob_range=1)
    pred2 = model(sample_input)
    w2 = model.get_weights()

    print("Predictions all close: ", np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))
    print("Average difference between predictions: ", (pred1 - pred2).mean())
    print("Average difference between weights : ", (w1 - w2).abs().mean())
    print("Sample weights: ")
    print(w1[0:10])
    print(w2[0:10])

    print("Samples Predictions")
    print(pred1.flatten()[0:10])
    print(pred2.flatten()[0:10])

    model.grapher.plot()
