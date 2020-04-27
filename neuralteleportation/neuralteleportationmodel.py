import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.mergelayers import Concat, Add
from neuralteleportation.network_graph import NetworkGrapher
from neuralteleportation.layers.neuralteleportationlayers import NeuronLayerMixin, MergeLayersMixin


class NeuralTeleportationModel(nn.Module):
    def __init__(self, network, input_shape):
        super(NeuralTeleportationModel, self).__init__()
        sample_input = torch.rand(input_shape)
        self.net = network
        self.grapher = NetworkGrapher(network, sample_input)
        self.graph = self.grapher.get_graph()

    def forward(self, x):
        return self.net(x)

    def get_change_of_basis(self, basis_range=10):
        """
          Returns list of np.arrays of change of basis per neuron
        """
        # self.graph[0]['prev_cob'] = self.graph[0]['module'].get_input_cob()

        current_cob = None  # Cob for last neuron layer to be applied to non-neuron layers ie. Activations
        next_cob = None  # Cob obtained from residual connections to be applied to flowing neuron-layers

        for i, layer in enumerate(self.graph):
            if isinstance(layer['module'], NeuronLayerMixin):
                # Check if this is the last neuron layer
                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    # print("LAST NEURON LAYER")
                    # if next_cob is not None:
                    #     raise ValueError("Last layer cannot be connected to previous layer, it must be ones")
                    current_cob = layer['module'].get_output_cob()
                elif not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[:i]]).any():
                    # print("First NEURON LAYER")
                    initial_cob = layer['module'].get_input_cob()
                    layer['prev_cob'] = initial_cob
                    for l in self.graph[:i]:
                        l['prev_cob'] = initial_cob
                        l['cob'] = initial_cob
                    current_cob = layer['module'].get_cob(basis_range=basis_range)

                # # Check  if there was a residual connection
                elif next_cob is not None:
                    # print("Next COB")
                    # current_cob = next_cob
                    current_cob = layer['module'].get_cob(basis_range=basis_range)
                    next_cob = None
                else:
                    current_cob = layer['module'].get_cob(basis_range=basis_range)

            if isinstance(layer['module'], Add):
                '''If the layer is an addition layer, two operations must occur
                    * Get the cob from the residual connection -> residual_cob.
                    * Get the cob from the previous layer -> prev_cob
                    1. The addition layer must scale the input -> y = x1 + x2*residual_cob/prev_cob
                    2. The next neuron layer must have the same cob as the residual_cob
                '''
                # print("ADD")
                # TODO GET LAYER THAT IS NOT i-1
                # connection_layer_index = layer['in'][0]
                # connection_layer_index = next((x for x in layer['in'] if x != i-1), None)
                connection_layer_index = min(layer['in'])
                # print('connection_layer_index: ', connection_layer_index)

                if not np.array([isinstance(l['module'], NeuronLayerMixin) for l in self.graph[i + 1:]]).any():
                    '''If there is no layer after, previous layer must be ones. '''
                    raise ValueError("NOT SUPPORTED YET: Must have neuron layer after residual connection")
                    # current_cob = self.graph[connection_layer_index - 1]['module'].get_output_cob()
                    # self.graph[connection_layer_index - 1]['cob'] = current_cob
                    # print(self.graph[connection_layer_index - 1]['module'])
                    # print("ADD IS LAST LAYER")
                # else:
                # print(connection_layer_index)

                next_cob = self.graph[connection_layer_index]['cob']
                # print(self.graph[connection_layer_index]['module'])
                # print(next_cob)
                current_cob = next_cob
                # current_cob = self.graph[connection_layer_index - 1]['cob']

            if isinstance(layer['module'], Concat):
                '''If the layer is concatenation the change of basis for this layer is the concatenation of all change
                   of basis of previous connected layers.
                '''
                # print("Concat")
                previous_layer_indexes = self.graph[i]['in']
                # print(previous_layer_indexes)
                current_cob = np.concatenate([self.graph[j]['cob'] for j in previous_layer_indexes])
                # print(current_cob.shape)

            if i > 0:
                # # TODO make constant for all layers, look for layer i-1
                # if isinstance(layer['module'], Add):
                #     input_layer_index = self.graph[i]['in'][1]  # Get previous layer to apply to input.
                #     # layer['prev_cob'] = self.graph[i-1]['cob']
                # else:
                #     input_layer_index = self.graph[i]['in'][0]  # only merge layers should have two inputs
                # print(i, ', ', layer['module'])
                # input_layer_index = next((x for x in layer['in'] if x == i - 1), layer['in'][0]) # if multiple inputs, get input that is i-1 else get the first input
                input_layer_index = max(layer['in']) # if multiple inputs, get input that is i-1 else get the first input
                layer['prev_cob'] = self.graph[input_layer_index]['cob']

            layer['cob'] = current_cob

    def teleport(self, cob_range=10):
        """
          Applies change of basis to each of the linear layer weights
        """
        self.get_change_of_basis(cob_range)

        # print(self.graph)

        for k, layer in enumerate(self.graph):
            # print(k, ',', layer['module'].__class__.__name__, " , ", layer['prev_cob'].shape, ', ', layer['cob'].shape)
            # print(k, ',', layer['module'].__class__.__name__, " , ", layer['prev_cob'], ', ', layer['cob'])
            layer['module'].apply_cob(prev_cob=layer['prev_cob'], next_cob=layer['cob'])

    def reset_weights(self):
        """Reset all layers."""

        def reset(m):
            for m in self.modules():
                getattr(m, 'reset_parameters', lambda: None)()

        self.apply(reset)

    def get_neuron_layers(self):
        # print(self.grapher.ordered_layers)
        # print([l for l in self.grapher.ordered_layers if isinstance(l, NeuronLayerMixin)])
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
    from neuralteleportation.models.test_models.test_models import *
    from neuralteleportation.models.test_models.residual_models import *
    from neuralteleportation.models.test_models.dense_models import *
    import random
    import torchvision.models as models

    # seed = 0
    #
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)

    # model = CombinedModule()
    # model = SplitConcatModel()
    # model = DenseNet2()
    # model = ResidualNet2()
    # model = ResidualNet5()

    # model = ConvTransposeNet()
    model = SplitConcatModel()
    sample_input_shape = (1, 1, 28, 28)

    # model = UNet(input_channels=1, output_channels=4, bilinear=False)
    # sample_input_shape = (1, 1, 256, 256)

    # model = resnet18(pretrained=False)
    # sample_input_shape = (1, 3, 224, 224)

    sample_input = torch.rand(sample_input_shape)
    model = NeuralTeleportationModel(network=model, input_shape=sample_input_shape)
    # try:
    # summary(model, sample_input_shape[1:])
    # except:
    #     print("SUMMARY FAILED")

    model.get_change_of_basis()

    # print(model)

    pred1 = model(sample_input)
    w1 = model.get_weights()
    # print(model(sample_input))
    print(model(sample_input).shape)
    model.teleport(cob_range=1000)
    pred2 = model(sample_input)
    # print(model(sample_input))
    print(model(sample_input).shape)
    w2 = model.get_weights()

    print(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))
    print((pred1 - pred2).mean())
    # print((pred1 - pred2))
    print((w1 - w2).abs().mean())
    print(w1[0:10])
    print(w2[0:10])

    print("Prediction")
    print(pred1.flatten()[0:10])
    print(pred2.flatten()[0:10])

    print(model.get_weights().shape)

    model.grapher.plot()
