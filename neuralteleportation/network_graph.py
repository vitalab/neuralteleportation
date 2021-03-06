from _collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from packaging import version


class NetworkGrapher:
    """
        This class computes the graph of a given network using torch.jit.
        The graph is essential for teleporting networks containing residual and dense layers.

    Args:
        network (nn.Module): network to be graphed.
        sample_input (torch.Tensor): Sample input for the network. (Required for  torch.jit.Trace(.))
    """

    def __init__(self, network: nn.Module, sample_input: torch.Tensor):
        self.network = network
        self.sample_input = sample_input
        self.ordered_layers = self.get_ordered_layers()
        self.graph = self.get_graph()

    def get_ordered_layers(self):
        """
            Get layer from self.network in the order of the forward pass.

        Returns:
            List of nn.Modules
        """

        def get_layers(model):
            layers = []
            for layer in model.children():
                if len(list(layer.children())) > 0:
                    layers.extend(get_layers(layer))
                else:
                    layers.append(layer)

            return layers

        all_layers = get_layers(self.network)

        ordered_layers = []

        def add_layer_hook(self, input, output):
            ordered_layers.append(self)

        handles = []
        for layer in all_layers:
            handles.append(layer.register_forward_hook(add_layer_hook))

        _ = self.network(self.sample_input)

        for handle in handles:
            handle.remove()

        return ordered_layers

    @staticmethod
    def get_graph_from_trace(trace):
        """
            Extract the graph from the trace according to the Pytorch version.

        Args:
            trace: trace obtained from torch.jit.trace

        Returns:
            trace.graph or trace.inlined_graph
        """
        if version.parse(torch.__version__) <= version.parse('1.3.1'):
            return trace.graph
        else:
            return trace.inlined_graph

    def get_graph(self):
        """
            Use torch.jit to get the graph of the network.

        Returns:
            network graph : [{'in': [], 'idx': [], 'out' = [], 'module' nn.Module}, ... ]
        """

        trace = torch.jit.trace(self.network, (self.sample_input,))

        graph = self.extract_pytorch_graph(self.get_graph_from_trace(trace))

        # remove empty keys
        graph = {k: v for k, v in graph.items() if k}

        # remove extra inputs
        graph = self.remove_inputs(graph)

        # remove extra outputs
        graph = self.remove_outputs(graph)

        # correct indexes
        graph = self.correct_indexes(graph)

        if len(graph) != len(self.ordered_layers):
            raise ValueError("Length of layer dict must be same length as ordered layers from model."
                             "Probably because operations were used in model.forward."
                             "Change these operations to nn.Module."
                             "OR because same layer was used more than once!")

        graph = self.reformat_layers(graph, self.ordered_layers)

        return graph

    @staticmethod
    def extract_pytorch_graph(inlined_graph) -> Dict:
        """
            Create a dict for the graph generated by torch.jit.trace.
        Args:
            inlined_graph: graph generated by torch.jit

        Returns:
            Dict containing layers and their connections.

        """

        layers = defaultdict(lambda: defaultdict(list))

        for i, node in enumerate(inlined_graph.nodes()):
            layers[node.scopeName()]['in'].extend([i.unique() for i in node.inputs()])
            layers[node.scopeName()]['out'].extend([i.unique() for i in node.outputs()])

        # Remove indexes that are both in input and output
        for k in layers.keys():
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
    def remove_inputs(layers: Dict) -> Dict:
        """
        Remove inputs that are not outputs of other layers. (These will be constants and other parameters)
        """

        # Get all output indexes
        layer_outputs = []
        for k in layers.keys():
            layer_outputs.extend(layers[k]['out'])

        # Remove inputs if not in layer_outputs
        for k in layers.keys():
            layers[k]['in'] = [x for x in layers[k]['in'] if x in layer_outputs]

        return layers

    @staticmethod
    def remove_outputs(layers: Dict) -> Dict:
        """
        Remove outputs that are not inputs of other layers. (These will be constants and other parameters)
        """

        # Get all input indexes
        layer_inputs = []
        for i, k in enumerate(layers.keys()):
            layer_inputs.extend(layers[k]['in'])

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
    def correct_indexes(layers: Dict) -> Dict:
        """
            Correct the layer indexes to start at zero.
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

    @staticmethod
    def reformat_layers(layers, module_list):
        """
        This method reformats the graph dict in the following format
            graph  = [{'in': [], 'idx': [], 'out' = [], 'module' nn.Module}, ... ]
        Each layer is a dict containing the following keys: in, idx, out and module.

        Args:
            layers (Dict): Refined dict obtained from torch.jit.trace.
            module_list (List): List of ordered modules in the network.

        Returns:
            list of dicts representing the model graph
        """
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

    def plot(self, block=True):
        import networkx as nx

        labels = {i: l['module'].__class__.__name__ for i, l in enumerate(self.graph)}

        G = nx.MultiDiGraph()
        pos = {}
        y = 1
        x = 1
        for i, l in enumerate(self.graph):
            G.add_node(i)
            pos[i] = [x, y]
            x += 10

        for i, l in enumerate(self.graph):
            for j in l['out']:
                G.add_edge(i, j)

        nx.draw_networkx_labels(G, pos, labels, font_size=5)
        nx.draw_networkx_nodes(G, pos, node_color='y', node_size=100)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, connectionstyle='arc3, rad=0.5')
        plt.show(block=block)

    def print_graph(self):
        print("--------------------------Print graph--------------------------------")
        f = "{:40}   {} -> {} -> {}"
        print(f.format("Module", "inputs", "index", "outputs"))
        for layer in self.graph:
            print(f.format(
                layer['module'].__class__.__name__,
                layer['in'],
                layer['idx'],
                layer['out']))

    def print_jit_graph(self):
        """
            List all the nodes in a PyTorch graph.
            This prints the function output used for generating the original graph dict.

        """

        trace = torch.jit.trace(self.network, (self.sample_input,))
        graph = self.get_graph_from_trace(trace)

        f = "{} {:25} {:40}   {} -> {}"
        print(f.format("index", "kind", "scopeName", "inputs", "outputs"))

        for i, node in enumerate(graph.nodes()):
            print(f.format(i, node.kind(), node.scopeName(),
                           [i.unique() for i in node.inputs()],
                           [i.unique() for i in node.outputs()]
                           ))


if __name__ == '__main__':
    from torchsummary import summary
    from neuralteleportation.models.generic_models.residual_models import ResidualNet2

    model = ResidualNet2()
    x = torch.rand((1, 1, 28, 28))
    summary(model, (1, 28, 28), device='cpu')

    grapher = NetworkGrapher(model, x)

    grapher.print_jit_graph()

    grapher.print_graph()

    grapher.plot()
