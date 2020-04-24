import numpy as np
import torch
import torch.nn as nn
from neuralteleportation.layers.network_graph import NetworkGrapher
from neuralteleportation.layers.layers_v3 import NeuronLayerMixin, MergeLayersMixin


class NeuralTeleportationModel(nn.Module):
    def __init__(self, network, sample_input):
        super(NeuralTeleportationModel, self).__init__()
        self.net = network
        self.grapher = NetworkGrapher(model, sample_input)
        self.graph = self.grapher.get_graph()

    def forward(self, x):
        return self.net(x)

    def get_change_of_basis(self, basis_range=10):
        """
          Returns list of np.arrays of change of basis per neuron
        """
        self.graph[0]['prev_cob'] = self.graph[0]['module'].get_input_cob()

        current_cob = None  # Cob for last neuron layer to be applied to non-neuron layers ie. Activations
        next_cob = None  # Cob obtained from residual connections to be applied to flowing neuron-layers

        for i, layer in enumerate(self.graph):
            if isinstance(layer['module'], NeuronLayerMixin):
                # Check if this is the last neuron layer
                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    print("LAST NEURON LAYER")
                    if next_cob is not None:
                        raise ValueError("Last layer cannot be connected to previous layer, it must be ones")
                    current_cob = layer['module'].get_output_cob()

                # Check  if there was a residual connection
                elif next_cob is not None:
                    print("Next COB")
                    current_cob = next_cob
                    next_cob = None
                else:
                    current_cob = layer['module'].get_cob()

            if isinstance(layer['module'], Add):
                '''If the layer is an addition layer, two operations must occur
                    * Get the cob from the residual connection -> residual_cob.
                    * Get the cob from the previous layer -> prev_cob
                    1. The addition layer must scale the input -> y = x1 + x2*residual_cob/prev_cob
                    2. The next neuron layer must have the same cob as the residual_cob
                '''
                print("ADD")
                connection_layer_index = layer['in'][0]
                print(connection_layer_index)
                next_cob = self.graph[connection_layer_index - 1]['cob']
                print(next_cob)
                current_cob = next_cob

            if isinstance(layer['module'], Concat):
                '''If the layer is concatenation the change of basis for this layer is the concatenation of all change
                   of basis of previous connected layers.
                '''
                print("Concat")
                previous_layer_indexes = self.graph[i]['in']
                print(previous_layer_indexes)
                current_cob = np.concatenate([self.graph[j]['cob'] for j in previous_layer_indexes])
                print(current_cob.shape)

            if i > 0:
                if isinstance(layer['module'], Add):
                    input_layer_index = self.graph[i]['in'][1]  # Get previous layer to apply to input.
                    # layer['prev_cob'] = self.graph[i-1]['cob']
                else:
                    input_layer_index = self.graph[i]['in'][0]  # only merge layers should have two inputs
                layer['prev_cob'] = self.graph[input_layer_index]['cob']

            layer['cob'] = current_cob

    def teleport(self, cob_range=10):
        """
          Applies change of basis to each of the linear layer weights
        """
        # cob = self.get_change_of_basis(cob_range)

        for k, layer in enumerate(self.graph):
            print(k, ',', layer['module'].__class__.__name__, " , ", layer['prev_cob'], ', ', layer['cob'])
            layer['module'].apply_cob(prev_cob=layer['prev_cob'], next_cob=layer['cob'])

    def reset_weights(self):
        """Reset all layers."""

        def reset(m):
            for m in self.modules():
                getattr(m, 'reset_parameters', lambda: None)()

        self.apply(reset)

    def get_neuron_layers(self):
        return [l for l in self.grapher.ordered_layers if isinstance(l, NeuronLayerMixin)]

    def get_weights(self, flat: bool = True):
        """
            Return model weights
        Args:
            flat: bool, if true weights are returned as flatten torch tensor

        Returns:
            torch.Tensor or list containing model weights

        """
        w = []

        for k, layer in enumerate(self.get_neuron_layers()):
            w.extend(layer.get_weights())

        if flat:
            return torch.cat(w)
        else:
            return w

    def set_weights(self, weights):
        counter = 0

        for k, layer in enumerate(self.get_neuron_layers()):
            nb_params = layer.get_nb_params()
            w = weights[counter:counter + nb_params]
            layer.set_weights(w)
            counter += nb_params

    def get_grad(self, data, target, loss_fn, flat=True):
        grad = []
        output = self.net(data)
        loss = loss_fn(output, target)
        loss.backward()

        for k, layer in enumerate(self.get_neuron_layers()):
            grad.append(layer.weight.grad.flatten())

            if layer.bias is not None:
                grad.append(layer.bias.grad.flatten())

        if flat:
            return torch.cat(grad)
        else:
            return grad


if __name__ == '__main__':
    from torchsummary import summary
    from neuralteleportation.layers.test_models import *
    import random
    import torchvision.models as models

    # seed = 0
    #
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)

    # model = CombinedModule()
    # model = SplitConcatModel()
    model = ResidualNet()
    sample_input = torch.rand((1, 1, 28, 28))
    model = NeuralTeleportationModel(network=model, sample_input=sample_input)
    # summary(model, (1, 28, 28))

    # model.grapher.plot()

    model.get_change_of_basis()

    # print(model)
    x = torch.rand((1, 1, 28, 28))
    pred1 = model(x)
    w1 = model.get_weights()
    print(model(x))
    print(model(x).shape)
    model.teleport()
    pred2 = model(x)
    print(model(x))
    print(model(x).shape)
    w2 = model.get_weights()

    print(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))
    print((pred1 - pred2).sum())
    print((w1 - w2).sum())
