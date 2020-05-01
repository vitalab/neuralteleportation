from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.modules import Flatten
from _collections import defaultdict
import numpy as np

from matplotlib import pyplot as plt
from torchvision.models import ResNet


class Concat(nn.Module):
    def forward(self, input1, input2):
        return torch.cat([input1, input2], dim=1)


class Add(nn.Module):
    def forward(self, input1, input2):
        return input1 + input2


class NetworkGrapher:
    def __init__(self, model: nn.Module, example_input):
        self.model = model
        self.example_input = example_input
        self.ordered_layers = self.get_ordered_layers()
        self.graph = self.get_graph()

    def get_ordered_layers(self):

        def get_layers(model):
            layers = []
            # print("{} : {}".format(model, list(model.children())))
            for layer in model.children():
                if len(list(layer.children())) > 0:
                    layers.extend(get_layers(layer))
                else:
                    # print("append: {}".format(layer))
                    layers.append(layer)

            return layers

        all_layers = get_layers(self.model)

        ordered_layers = []

        def add_layer_hook(self, input, output):
            ordered_layers.append(self)
            # print(self)
            # print("Input: {}".format(input))
            # for i in input:
            #     print(i.grad_fn)
            # print("Ouput: {}".format(output))

        handles = []
        for layer in all_layers:
            handles.append(layer.register_forward_hook(add_layer_hook))

        out = self.model(self.example_input)

        for handle in handles:
            handle.remove()

        return ordered_layers

    def get_graph(self):
        trace, out = torch.jit.get_trace_graph(self.model, self.example_input)

        layers = self.dump_pytorch_graph(trace.graph())

        # remove empty keys
        layers = {k: v for k, v in layers.items() if k}

        # remove extra inputs
        layers = self.remove_inputs(layers)

        # remove extra outputs
        layers = self.remove_outputs(layers)

        # #correct indexes
        layers = self.correct_indexes(layers)
        # print(len(layers))
        # print(layers)

        # print("Reformat")
        if len(layers) != len(self.ordered_layers):
            raise ValueError("Length of layer dict must be same length as ordered layers from model."
                             "Probably because operations were used in model.forward."
                             "Change these operations to nn.Module."
                             "OR because same layer was used more than once!")

        layers = self.reformat_layers(layers, self.ordered_layers)
        # print(layers)
        # self.print_graph(layers)

        # print(len(layers))

        return layers

    @staticmethod
    def print_graph(graph):
        f = "{:40}   {} -> {} -> {}"
        print(f.format("Module", "inputs", "index", "outputs"))
        for layer in graph:
            print(f.format(
                layer['module'].__class__.__name__,
                layer['in'],
                layer['idx'],
                layer['out']))

    @staticmethod
    def reformat_layers(layers, module_list):
        new_layers = []

        for i, k in enumerate(layers):
            outputs = []
            for j, q in enumerate(layers):
                if i in layers[q]['in']:
                    outputs.append(j)

            new_layers.append({'idx': i,
                               'out': outputs,
                               'module': module_list[i]})

        for i, k in enumerate(new_layers):
            inputs = []
            for j, q in enumerate(new_layers):
                if i in new_layers[j]['out']:
                    inputs.append(j)
            new_layers[i]['in'] = inputs

        return new_layers

    @staticmethod
    def dump_pytorch_graph(graph):
        """List all the nodes in a PyTorch graph."""
        f = "{} {:25} {:40}   {} -> {}"
        # print(f.format("index", "kind", "scopeName", "inputs", "outputs"))
        # layers = {}
        layers = defaultdict(lambda: defaultdict(list))
        for i, node in enumerate(graph.nodes()):
            # if node.kind().split(':')[0] == 'aten':
            # print(f.format(i, node.kind(), node.scopeName(),
            #                [i.unique() for i in node.inputs()],
            #                [i.unique() for i in node.outputs()]
            #                ))
            # Only to last item with scopeName (this will be the output of the layer)
            # Other items are intermediary item ie. constants...
            layers[node.scopeName()]['in'].extend([i.unique() for i in node.inputs()])
            layers[node.scopeName()]['out'].extend([i.unique() for i in node.outputs()])

        # Remove indexes that are both in input and output
        for k in layers.keys():
            # print(k)
            # print(layers[k]['out'])
            # print(layers[k]['in'])
            for i in layers[k]['out'][:]:
                if i in layers[k]['in']:
                    # print(i)
                    layers[k]['out'].remove(i)
                    layers[k]['in'].remove(i)

            layers[k] = dict(layers[k])
            # Remove duplicates in list
            layers[k]['in'] = list(set(layers[k]['in']))
            layers[k]['out'] = list(set(layers[k]['out']))

        return layers

    @staticmethod
    def remove_inputs(layers: Dict):
        """
        Remove inputs that are not outputs of other layers. (These will be constants and other parameters)
        """

        # Get all out indexes
        layer_outputs = []
        for k in layers.keys():
            layer_outputs.extend(layers[k]['out'])

        # print("Layer outputs: {}".format(layer_outputs))

        # Remove inputs if not in layer_outputs
        for k in layers.keys():
            layers[k]['in'] = [x for x in layers[k]['in'] if x in layer_outputs]

        return layers

    @staticmethod
    def remove_outputs(layers: Dict):
        """
        Remove ouputs that are not outputs of other layers. (These will be constants and other parameters)
        """

        # Get all out indexes
        layer_inputs = []
        for i, k in enumerate(layers.keys()):
            layer_inputs.extend(layers[k]['in'])

        # print("Layer inputs: {}".format(layer_inputs))

        # Remove inputs if not in layer_outputs
        for k in layers.keys():
            layers[k]['out'] = [x for x in layers[k]['out'] if x in layer_inputs]

        # Remove layers if there are not outputs (except for last layer)
        new_layers = {}
        for i, k in enumerate(layers.keys()):
            if not (len(layers[k]['out']) == 0 and i != len(layers.keys()) - 1):
                new_layers[k] = layers[k]

        return new_layers

    @staticmethod
    def correct_indexes(layers):
        """
            Correct the layer indexes to start at one.
        """
        counter = 0
        correction_dict = {}
        for k in layers.keys():
            l_in = layers[k]['in']
            for i in range(len(l_in)):
                if l_in[i] in correction_dict.keys():
                    l_in[i] = correction_dict[l_in[i]]
                else:
                    correction_dict[l_in[i]] = counter
                    l_in[i] = counter
                    counter += 1
            l_out = layers[k]['out']
            for i in range(len(l_out)):
                if l_out[i] in correction_dict.keys():
                    l_out[i] = correction_dict[l_out[i]]
                else:
                    correction_dict[l_out[i]] = counter
                    l_out[i] = counter
                    counter += 1

        return layers

    def plot(self, block=True):
        import networkx as nx
        layers = self.get_graph()

        labels = {i: l['module'].__class__.__name__ for i, l in enumerate(layers)}
        # print(labels)

        G = nx.MultiDiGraph()
        pos = {}
        y = 1
        x = 1
        for i, l in enumerate(layers):
            G.add_node(i)
            pos[i] = [x, y]
            x += 10

        for i, l in enumerate(layers):
            for j in l['out']:
                G.add_edge(i, j)

        nx.draw_networkx_labels(G, pos, labels, font_size=5)
        nx.draw_networkx_nodes(G, pos, node_color='y', node_size=100)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, connectionstyle='arc3, rad=0.5')
        plt.show(block=block)



if __name__ == '__main__':
    from torchsummary import summary
    from neuralteleportation.models.test_models.test_models import ResidualNet
    # model = models.resnet18()
    # x = torch.rand((1, 3, 224, 224))
    # summary(model, (3, 224, 224), device='cpu')

    # model = Net()
    # model = DenseNet()
    model = ResidualNet()
    x = torch.rand((1, 1, 28, 28))
    summary(model, (1, 28, 28), device='cpu')

    grapher = NetworkGrapher(model, x)
    grapher.get_graph()

    print("Ordered layers")
    print(grapher.ordered_layers)
    print(len(grapher.ordered_layers))

    grapher.plot()
