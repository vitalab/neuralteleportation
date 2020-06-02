from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.changeofbasisutils import get_random_cob
from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin, COBForwardMixin

from matplotlib import pyplot as plt


class NeuronLayerMixin(NeuralTeleportationLayerMixin):
    in_features: int
    out_features: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create new tensor for weight and bias to allow gradient to be computed with respect to cob.
        self.w = self.weight.clone().detach().requires_grad_(True)
        if self.bias is not None:
            self.b = self.bias.clone().detach().requires_grad_(True)

    def get_cob(self, basis_range: int = 0.5, sampling_type='usual') -> torch.Tensor:
        """Returns a random change of basis for the output features of the layer.

        Args:
            basis_range: range for the change of basis.

        Returns:
            change of basis.
        """
        return get_random_cob(range_cob=basis_range, size=self.out_features, sampling_type=sampling_type)

    def get_output_cob(self) -> torch.Tensor:
        """Get change of basis for the layer if it is an output layer.

        Returns:
            Ones of size of the output features.
        """
        return torch.ones(self.out_features)

    def get_input_cob(self) -> torch.Tensor:
        """Get change of basis for the layer if it is an input layer.

        Returns:
            Ones of size of the input features.
        """
        return torch.ones(self.in_features)

    def get_nb_params(self) -> int:
        """Get the number of parameters in the layer (weight and bias).

        Returns:
            number of parameters in the layer.
        """
        nb_params = np.prod(self.weight.shape)
        if self.bias is not None:
            nb_params += np.prod(self.bias.shape)

        return nb_params

    def get_weights(self, flatten=True, bias=True) -> Tuple[torch.Tensor, ...]:
        """Get the weights from the layer.

        Returns:
            tuple of weight tensors.
        """
        if self.bias is not None and bias:
            if flatten:
                return self.w.flatten(), self.b.flatten()
            else:
                return self.w, self.b
        else:
            if flatten:
                return self.w.flatten(),
            else:
                return self.w,

        # if self.bias is not None and bias:
        #     if flatten:
        #         return self.weight.flatten(), self.bias.flatten()
        #     else:
        #         return self.weight, self.bias
        # else:
        #     if flatten:
        #         return self.weight.flatten(),
        #     else:
        #         return self.weight,

    def set_weights(self, weights: torch.Tensor):
        """Set weights for the layer.

        Args:
            weights: weights to apply to the model.
        """
        counter = 0
        w_shape = self.weight.shape
        w_nb_params = np.prod(w_shape)
        self.w = weights[counter:counter + w_nb_params].reshape(w_shape)
        self.weight = torch.nn.Parameter(self.w, requires_grad=True)
        counter += w_nb_params

        if self.bias is not None:
            b_shape = self.bias.shape
            b_nb_params = np.prod(b_shape)
            self.b = weights[counter:counter + b_nb_params].reshape(b_shape)
            self.bias = torch.nn.Parameter(self.b, requires_grad=True)

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        print("Apply COB")
        print(self.weight)
        self.w = self.weight * self._get_cob_weight_factor(prev_cob, next_cob).type_as(self.weight)
        self.weight = nn.Parameter(self.w, requires_grad=True)
        print(self.weight)
        if self.bias is not None:
            self.b = self.bias * next_cob.type_as(self.bias)
            print(self.bias)
            self.bias = torch.nn.Parameter(self.b, requires_grad=True)
            print(self.bias)

    def _get_cob_weight_factor(self, prev_cob: torch.Tensor, next_cob: torch.Tensor) -> torch.Tensor:
        """Computes the factor to apply to the weights of the current layer to perform a change of basis.

        Args:
            prev_cob: change of basis of the previous layer.
            next_cob: change of basis of the following layer.

        Returns:
            factors to apply to the weights of the current layer to apply the change of basis.
        """
        raise NotImplementedError

    @staticmethod
    def calculate_cob(weights, target_weights, prev_cob, concat=True):
        """
        Compute the cob to teleport from the current weights to the target_weights
        """
        raise NotImplementedError

    @staticmethod
    def calculate_last_cob(initial_weights1, initial_weights2, target_weights1, target_weights2, prev_cob):
        """
        Compute the cob to teleport from the current weights to the target_weight for the last cob.
        """
        raise NotImplementedError


class LinearCOB(NeuronLayerMixin, nn.Linear):

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        if len(prev_cob) != self.in_features:  # if previous layer is Conv2D, duplicate cob for each feature map.
            feature_map_size = self.in_features // len(prev_cob)  # size of feature maps
            cob = []
            for i, c in enumerate(prev_cob):
                cob.extend(np.repeat(c, repeats=feature_map_size).tolist())
            prev_cob = np.array(cob)

        super().apply_cob(prev_cob, next_cob)

    def _get_cob_weight_factor(self, prev_cob: torch.Tensor, next_cob: torch.Tensor) -> torch.Tensor:
        return next_cob[..., None] / prev_cob[None, ...]

    @staticmethod
    def calculate_cob(weights, target_weights, prev_cob, concat=True):
        """
        Compute the cob to teleport from the current weights to the target_weights
        """
        cob = []
        for (wi, wi_hat) in zip(weights, target_weights):
            ti = (wi / prev_cob).dot(wi_hat) / (wi / prev_cob).dot(wi / prev_cob)
            cob.append(ti)

        return torch.tensor(cob)

    @staticmethod
    def calculate_last_cob(initial_weights1, target_weights1, initial_weights2, target_weights2, prev_cob):
        t = []
        for i in range(initial_weights1.shape[0]):
            w0 = initial_weights1[i, :]
            w0_hat = target_weights1[i, :]
            w1 = initial_weights2[:, i]
            w1_hat = target_weights2[:, i]
            t0 = prev_cob
            t2 = torch.ones(target_weights2.shape[0])

            # print(w0.shape)
            # print(w1.shape)
            # print(w0_hat.shape)
            # print(w1_hat.shape)
            # print(t0.shape)
            # print(t2.shape)

            ti = torch.tensor(1.0)

            eta = 0.1

            grads = []
            for _ in range(200):
                grad = (2 * ti * (w0 / t0).dot(w0 / t0) -
                        2 * (w0 / t0).dot(w0_hat) -
                        2 * torch.pow(ti, -3) * (w1 * t2).dot(w1 * t2) +
                        2 * torch.pow(ti, -2) * (w1 * t2).dot(w1_hat))
                # print("ti: ", ti)
                # print("grad: ", grad)
                ti = ti - eta * grad
                grads.append(grad.item())

            # plt.figure()
            # plt.plot(grads)
            # plt.show()

            # print("final ti: ", ti)
            t.append(ti)

        return torch.tensor(t)


class ConvMixin(NeuronLayerMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = self.in_channels
        self.out_features = self.out_channels


class Conv2dCOB(ConvMixin, nn.Conv2d):

    def _get_cob_weight_factor(self, prev_cob: torch.Tensor, next_cob: torch.Tensor) -> torch.Tensor:
        return (next_cob[..., None] / prev_cob[None, ...])[..., None, None]


class ConvTranspose2dCOB(ConvMixin, nn.ConvTranspose2d):

    def _get_cob_weight_factor(self, prev_cob: torch.Tensor, next_cob: torch.Tensor) -> torch.Tensor:
        return (next_cob[None, ...] / prev_cob[..., None])[..., None, None]


class BatchNormMixin(COBForwardMixin, NeuronLayerMixin):
    cob_field = 'prev_cob'

    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.in_features = self.out_features = self.num_features

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        self.prev_cob = prev_cob
        super().apply_cob(prev_cob, next_cob)

    def _get_cob_weight_factor(self, prev_cob: np.ndarray, next_cob: np.ndarray) -> np.ndarray:
        return next_cob

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.base_layer().forward(self, input / self.prev_cob)


class BatchNorm2dCOB(BatchNormMixin, nn.BatchNorm2d):
    reshape_cob = True


class BatchNorm1dCOB(BatchNormMixin, nn.BatchNorm1d):
    reshape_cob = True
