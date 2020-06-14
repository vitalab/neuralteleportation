import torch.nn as nn

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


class LayerHook:
    """
    This class serves as an anchor point for a specific layer to be able to catch the output on a forward pass.
    The created object is a wrapper for the RemovableHandle that torch.nn.register_forward_hook returns to better
    handle when the hook is active or inactive.

    Args:
        model, a NeuralTeleportationModel object
        state, Tuple (layer_name, hook) where the layer_name is a string and hook is a Callback with the signature (
        self, input, output)


    Ex:
        import torch
        import torch.nn as nn
        import LayerHook as LH

        def hook(self, input, output):
            print(output)

        net = ... (Any nn.Module)

        hook = LH(net,("conv1",hook))

        x = torch.ones((1,1,28,28))
        net(x)

    """

    def __init__(self, model, state=(None, None)):
        self.handle = None
        self.model = model
        if state[0] is None:
            self.layer = None
            self.hook = None
        else:
            self.set_hooked_layer(state[0], state[1])

    def disable_hook(self):
        self.handle.remove()

    def enable_hook(self):
        self.handle = self.layer.register_forward_hook(self.hook)

    def set_hooked_layer(self, layer_name, hook):
        """
        This is a function used to get output from a specific layer or layers inside a given model.

        Args:
            model, any nn.Module or NeuralTeleportationModel that can generate a layer graph or layer Ord. Dict.
            layer_id, the key or array pointer to the specific layer
            hook, a callback with the following signature (self, input, output)
        """
        self.hook = hook
        # Find the specific layer inside type-dependent model.
        if isinstance(self.model, NeuralTeleportationModel):
            for _, l in enumerate(self.model.grapher.network.named_modules()):
                if layer_name in l:
                    self.layer = l[1]
                    self.handle = l[1].register_forward_hook(hook)
                    break

        elif isinstance(self.model, nn.Module):
            for _, l in enumerate(self.model.named_modules()):
                if layer_name in l:
                    self.layer = l[1]
                    self.handle = l[1].register_forward_hook(hook)
                    break

        if self.handle is None:
            raise TypeError(
                "Handle is empty meaning that layer id was not found within the model.")

    def __del__(self):
        self.disable_hook()


if __name__ == "__main__":
    # This is the toy example in the Class description.
    import torch
    from neuralteleportation.models.generic_models.dense_models import DenseNet

    def hook(self, input, output):
        print(output)

    net = DenseNet()

    hook = LayerHook(net, ("conv1", hook))

    x = torch.ones((1, 3, 32, 32))
    net(x)
