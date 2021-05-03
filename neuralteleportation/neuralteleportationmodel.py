from typing import Tuple, Callable, Union, List

import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.changeofbasisutils import get_random_cob
from neuralteleportation.layers.neuron import NeuronLayerMixin
from neuralteleportation.network_graph import NetworkGrapher
from neuralteleportation.layers.merge import Add, Concat


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

        was_training = self.training
        self.eval()
        self.grapher = NetworkGrapher(network, sample_input)
        self.graph = self.grapher.get_graph()
        if was_training:
            self.train()

        self.initialize_cob()

    def forward(self, x):
        return self.network(x)

    def get_cob_size(self) -> int:
        """
            Get size of network's change of basis without input and output.

        Returns:
            (int) cob size
        """
        size = 0
        neuron_layers = self.get_neuron_layers()
        for i in range(len(neuron_layers) - 1):
            size += neuron_layers[i].out_features
        return size

    def initialize_cob(self) -> None:
        """ Set the cob to ones. """
        self.teleport_activations(torch.ones(self.get_cob_size()))

    def init_like_histogram(self, hist: np.ndarray, bin_edges: np.ndarray) -> None:
        """Initializes the weights of the network as if they were sampled from the provided histogram.

        Args:
            hist: Probability of each bin in the histogram.
            bin_edges: Boundaries of each bin (length(hist)+1).
        """
        cum_hist = np.cumsum(hist)
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2

        def init_layer_like_histogram(m):
            if isinstance(m, nn.Linear):
                for i in range(m.weight.shape[0]):
                    for j in range(m.weight.shape[1]):
                        rand_bin = (np.abs(cum_hist-np.random.rand(1))).argmin()
                        m.weight[i, j] = bin_centers[rand_bin].item()

        with torch.no_grad():
            self.apply(init_layer_like_histogram)

    def generate_random_cob(self, cob_range: float = 0.5, sampling_type: str = 'intra_landscape',
                            requires_grad: bool = False, center: float = 1) -> torch.Tensor:
        """
            Generate random cob with the correct size for the network.
        Args:
            cob_range (float): range_cob for the change of basis. Recommended between 0 and 1, but can take any
                                positive range_cob.
            sampling_type (str): label for type of sampling for change of basis
            requires_grad (bool): whether the cob tensor should require gradients
            center(float): the center for the change of basis sampling
        Returns:
            (Tensor) cob
        """
        return get_random_cob(range_cob=cob_range, size=self.get_cob_size(), requires_grad=requires_grad,
                              sampling_type=sampling_type, center=center)

    def undo_teleportation(self):
        """
            Undo the current teleportation.
        """
        # Undo teleportation for weights
        self.teleport_weights(1 / self.get_cob())
        # Set cob for activations to 1.
        self.initialize_cob()

    def random_teleport(self, cob_range: float = 0.5, sampling_type: str = 'intra_landscape',
                        reset_teleportation: bool = True, center: float = 1):
        """
          Applies random change of basis to each of the network layers.

        Returns:
            nn.Module of the network after teleportation
        """
        return self.teleport(self.generate_random_cob(cob_range, sampling_type, center=center),
                             reset_teleportation=reset_teleportation)

    def teleport(self, cob: torch.Tensor, reset_teleportation: bool = True):
        """
            Teleport the network.

        Args:
            cob (torch.Tensor): cob to teleport the network
            reset_teleportation (bool): if true, the previous teleportation is undone before teleporting with the cob

        Returns:
            neural teleportation model after teleportation
        """
        if reset_teleportation:
            self.undo_teleportation()
            self.teleport_weights(cob)
            self.teleport_activations(cob)
        else:
            previous_cob = self.get_cob()
            # Teleport weights
            self.teleport_weights(cob)
            # Set activation cob to product of previous cobs and new cob
            self.teleport_activations(previous_cob * cob)

        return self

    def teleport_weights(self, cob: torch.Tensor):
        """
            Teleport the network weights with the cob.
        """
        self.set_change_of_basis(cob)

        for k, layer in enumerate(self.graph):
            layer['module'].teleport(prev_cob=layer['prev_cob'], next_cob=layer['cob'])

    def teleport_activations(self, cob: torch.Tensor):
        """
            Teleport the network activations and non-weight layers with the cob.
        """
        self.set_change_of_basis(cob)

        for k, layer in enumerate(self.graph):
            layer['module'].apply_cob(prev_cob=layer['prev_cob'], next_cob=layer['cob'])

    def set_change_of_basis(self, cob: torch.Tensor, contains_ones: bool = False):
        """
            Set the change of basis for the network.

        Args:
            cob (torch.Tensor): change of basis to be applied to the model.
            contains_ones (bool): whether the cob contains input and output ones.
        """

        counter = 0
        current_cob = None  # Cob for last neuron layer to be applied following to non-neuron layers ie. Activations

        for i, layer in enumerate(self.graph):
            if isinstance(layer['module'], NeuronLayerMixin):
                # Check if this is the last neuron layer
                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    if contains_ones:
                        current_cob = cob[counter: counter + layer['module'].out_features]
                        assert torch.all(current_cob == 1), "Output layer cob values must be all ones."
                        counter += layer['module'].out_features
                    else:
                        current_cob = layer['module'].get_output_cob()

                # Check if this is the first neuron layer
                elif not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[:i]]).any():
                    if contains_ones:
                        initial_cob = cob[counter: counter + layer['module'].in_features]
                        assert torch.all(current_cob == 1), "Input layer cob values must be all ones."
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
                current_cob = torch.cat([self.graph[j]['cob'] for j in previous_layer_indexes])

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

    def get_weights(self, concat: bool = True, flatten=True, bias=True,
                    ignore_bn: bool = False, get_proxy_weight=False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
            Return model weights

        Args:
            concat (bool): if true weights are returned as concatenated torch tensor,
                            else in the form of a list of Tensors
            flatten (bool): if true weights are flattened
            bias (bool): if true bias is included
            get_proxy_weight (bool): if true, will only get the proxy weights without updating.

        Returns:
            torch.Tensor or list containing model weights

        """
        w = []

        for k, layer in enumerate(self.get_neuron_layers()):
            if ignore_bn and isinstance(layer, nn.BatchNorm2d):
                weight = layer.get_weights(flatten=flatten, bias=bias, get_proxy=get_proxy_weight)
                w.append(torch.zeros_like(weight[0]).flatten())
                if bias:
                    w.append(torch.zeros_like(weight[1]).flatten())
            w.extend(layer.get_weights(flatten=flatten, bias=bias, get_proxy=get_proxy_weight))

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

    def get_hessian(self, data: torch.Tensor, target: torch.Tensor, loss_fn: Callable, zero_grad: bool = True):
        if zero_grad:
            self.network.zero_grad()

        output = self.network(data)
        loss = loss_fn(output, target)
        grads = torch.autograd.grad(loss, self.network.parameters(), create_graph=True)
        grads = torch.cat([grad.flatten() for grad in grads])

        nb_params = len(grads)
        hessian = torch.empty((nb_params, nb_params))
        print(hessian.shape)

        # hessian = []
        for i, grad in enumerate(grads):
            h = torch.autograd.grad(grad, self.network.parameters(), create_graph=True)
            h = torch.cat([grad.flatten() for grad in h])
            hessian[i, :] = h

        # hessian = torch.cat(hessian)

        return hessian

    def get_hessian_trace(self, data: torch.Tensor, target: torch.Tensor, loss_fn: Callable, zero_grad: bool = True):
        if zero_grad:
            self.network.zero_grad()

        output = self.network(data)
        loss = loss_fn(output, target)
        grads = torch.autograd.grad(loss, self.network.parameters(), create_graph=True)
        grads = torch.cat([grad.flatten() for grad in grads])

        trace = 0

        for i, grad in enumerate(grads):
            h = torch.autograd.grad(grad, self.network.parameters(), create_graph=True)
            h = torch.cat([grad.flatten() for grad in h])
            trace += h[i]

        # hessian = torch.cat(hessian)

        return trace

    def get_cob(self, concat=True, contain_ones=False):
        """
            Get change of basis for network.

        Args:
            concat (bool): if true, cobs are concatenated into one tensor
            contain_ones (bool): if true, cob contains input and ouput cob of ones.

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

    def calculate_cob(self, initial_weights: List[torch.Tensor], target_weights: List[torch.Tensor],
                      concat: bool = True, eta: float = 0.1, steps: int = 1000):
        """
            Calculate the change of basis to teleport the initial weights to the target weights.
            Each cob is calculated individually following a closed form solution to
                        min_T ||T(initial_weights) - target_weights||


        Args:
            initial_weights (List[torch.Tensor]): initial weights on which teleportation is applied
            target_weights (List[torch.Tensor]): target weigths to obtain with teleportation.
            concat (bool): whether to concat cob into one tensor
            eta (float): learning rate for the gradient descent for the last cob optimisation
            steps (int): number of gradient descent steps for the last cob optimisation

        Returns:
            (torch.Tensor) calculated change of basis
        """
        layers = self.get_neuron_layers()

        cob = [layers[0].get_input_cob()]
        for i in range(len(layers) - 2):
            t = layers[i].calculate_cob(initial_weights[i], target_weights[i], cob[-1])
            cob.append(t)

        last_cob = layers[-2].calculate_last_cob(initial_weights[-2], target_weights[-2],
                                                 initial_weights[-1], target_weights[-1],
                                                 cob[-1], eta, steps)

        if last_cob is None:
            last_cob = layers[-2].calculate_cob(initial_weights[-2], target_weights[-2], cob[-1])

        cob.append(last_cob)

        cob.pop(0)  # Remove input layer cob.

        if concat:
            return torch.cat(cob)
        else:
            return cob

    def get_params(self):
        return self.get_weights(), self.get_cob()

    def set_params(self, weights, cob):
        self.teleport_activations(cob)
        self.set_weights(weights)

    def get_layer_hessians(self, data: torch.Tensor, target: torch.Tensor, loss_fn: Callable,
                           zero_grad: bool = True):
        if zero_grad:
            self.network.zero_grad()

        output = self.network(data)
        loss = loss_fn(output, target)
        grads = torch.autograd.grad(loss, self.network.parameters(), create_graph=True)
        grads = [grad.flatten() for grad in grads]

        hessians = []
        for i, grad in enumerate(grads):
            hessian = torch.empty((len(grad), len(grad)))
            for j, g in enumerate(grad):
                h = torch.autograd.grad(g, list(self.network.parameters())[i], create_graph=True)
                h = torch.cat([grad.flatten() for grad in h])
                hessian[j, :] = h
            print(grad.shape)
            print(list(self.network.parameters())[i].shape)
            hessians.append(hessian)

        return hessians

    def get_layer_hessian_traces(self, data: torch.Tensor, target: torch.Tensor, loss_fn: Callable,
                                 zero_grad: bool = True):
        if zero_grad:
            self.network.zero_grad()

        output = self.network(data)
        loss = loss_fn(output, target)
        grads = torch.autograd.grad(loss, self.network.parameters(), create_graph=True)
        grads = [grad.flatten() for grad in grads]

        traces = []
        for i, grad in enumerate(grads):
            trace = 0
            for j, g in enumerate(grad):
                h = torch.autograd.grad(g, list(self.network.parameters())[i], create_graph=True)
                h = torch.cat([grad.flatten() for grad in h])
                trace += h[j]
            traces.append(trace)

        return traces


if __name__ == '__main__':
    from tests.cobmodels_test import *
    from neuralteleportation.models.model_zoo.resnetcob import resnet18COB

    model = resnet18COB(pretrained=False, num_classes=10)
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
