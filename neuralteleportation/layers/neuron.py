from typing import Tuple

import numpy as np
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm

from neuralteleportation.changeofbasisutils import get_random_cob
from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin, COBForwardMixin


class NeuronLayerMixin(NeuralTeleportationLayerMixin):
    in_features: int
    out_features: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._set_proxy_weights()

    def _set_proxy_weights(self):
        """
            Create new tensor for weights and bias to allow gradient to be computed with respect to cob.
        """
        self.w = self.weight.clone().detach().requires_grad_()
        if self.bias is not None:
            self.b = self.bias.clone().detach().requires_grad_()

    def get_cob(self, basis_range: float = 0.5, sampling_type: str = 'intra_landscape',
                center: float = 1) -> torch.Tensor:
        """Returns a random change of basis for the output features of the layer.

        Args:
            basis_range (float): range for the change of basis.
            sampling_type(str): label for type of sampling for change of basis
            center(float): center for the sampling of the change of basis

        Returns:
            change of basis.
        """
        return get_random_cob(range_cob=basis_range, size=self.out_features, sampling_type=sampling_type,
                              center=center)

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

        return int(nb_params)

    def get_weights(self, flatten=True, bias=True, get_proxy: bool = True) -> Tuple[torch.Tensor, ...]:
        """Get the weights from the layer.

        Returns:
            tuple of weight tensors.
        """

        # Check if the weights were updated during training on loading weights.
        if get_proxy:
            self._set_proxy_weights()
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
        else:
            if self.bias is not None and bias:
                if flatten:
                    return self.weight.flatten(), self.bias.flatten()
                else:
                    return self.weight, self.bias
            else:
                if flatten:
                    return self.weight.flatten(),
                else:
                    return self.weight,

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

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        self.w = self.weight * self._get_cob_weight_factor(prev_cob, next_cob).type_as(self.weight)
        # This helps avoid a crash due to self.w not being a leaf variable,
        # meaning it is not at the begining of the graph and 
        # was created from an operation on other tensors (self.weight here).
        self.w = self.w.clone().detach().requires_grad_()
        self.weight = nn.Parameter(self.w, requires_grad=True)
        if self.bias is not None:
            self.b = self.bias * next_cob.type_as(self.bias)
            # Same as self.w
            self.b = self.b.clone().detach().requires_grad_()
            self.bias = torch.nn.Parameter(self.b, requires_grad=True)

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

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
    def calculate_cob(initial_weights, target_weights, prev_cob, ) -> torch.Tensor:
        """
        Compute the cob to teleport from the initial_weights to the target_weights.
        Using the closed form solution to:

                    min_T ||T(initial_weights) - target_weights||

        Args:
            initial_weights (torch.Tensor): initial weights on which teleportation is applied
            target_weights (torch.Tensor): target weigths to obtain with teleportation.
            prev_cob (torch.Tensor): Change of basis from the previous layer

        Returns:
            torch.Tensor, calculated cob

        """
        raise NotImplementedError

    @staticmethod
    def calculate_last_cob(initial_weights1, initial_weights2,
                           target_weights1, target_weights2, prev_cob, eta, steps) -> torch.Tensor:
        """
        Compute the cob to teleport from the initial_weights to the target_weight for the last cob considering that the
        output cob is always ones.

        prev_cob -- w1 --> cob -- w2 --> ones

        Args:
            initial_weights1 (torch.Tensor): layer n-1 initial weights on which teleportation is applied
            initial_weights2 (torch.Tensor): layer n initial weights on which teleportation is applied
            target_weights1 (torch.Tensor): layer n-1 target weigths to obtain with teleportation.
            target_weights2 (torch.Tensor): layer n target weigths to obtain with teleportation.
            prev_cob (torch.Tensor): Change of basis from the previous layer
            eta (float): learning rate for the gradient descent
            steps (int): number of gradient descent steps

        Returns:
            torch.Tensor, calculated cob

        """
        raise NotImplementedError


class LinearCOB(NeuronLayerMixin, nn.Linear):

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        if len(prev_cob) != self.in_features:  # if previous layer is Conv2D, duplicate cob for each feature map.
            feature_map_size = self.in_features // len(prev_cob)  # size of feature maps
            cob = []
            for i, c in enumerate(prev_cob):
                cob.append(c.repeat(feature_map_size))
            prev_cob = torch.cat(cob, dim=0)

        super().teleport(prev_cob, next_cob)

    def _get_cob_weight_factor(self, prev_cob: torch.Tensor, next_cob: torch.Tensor) -> torch.Tensor:
        return next_cob[..., None] / prev_cob[None, ...]

    @staticmethod
    def calculate_cob(weights, target_weights, prev_cob) -> torch.Tensor:
        """
        Compute the cob to teleport from the current weights to the target_weights
        """
        cob = []
        prev_cob = prev_cob.type_as(weights)
        for (wi, wi_hat) in zip(weights, target_weights):
            ti = (wi / prev_cob).dot(wi_hat) / (wi / prev_cob).dot(wi / prev_cob)
            cob.append(ti)

        return torch.tensor(cob)

    @staticmethod
    def calculate_last_cob(initial_weights1, target_weights1,
                           initial_weights2, target_weights2, prev_cob, eta, steps) -> torch.Tensor:
        t = []
        t0 = prev_cob.type_as(initial_weights1)
        t2 = torch.ones(target_weights2.shape[0]).type_as(initial_weights1)

        for i in tqdm(range(initial_weights1.shape[0])):
            w0 = initial_weights1[i, :]
            w0_hat = target_weights1[i, :]
            w1 = initial_weights2[:, i]
            w1_hat = target_weights2[:, i]

            ti = torch.tensor(1.0)

            grads = []

            for step in range(steps):
                grad = (2 * ti * (w0 / t0).dot(w0 / t0) -
                        2 * (w0 / t0).dot(w0_hat) -
                        2 * torch.pow(ti, -3) * (w1 * t2).dot(w1 * t2) +
                        2 * torch.pow(ti, -2) * (w1 * t2).dot(w1_hat))
                ti = ti - eta * grad.item()

                if torch.isnan(ti).any():
                    warnings.warn("Calculating last cob failed. Calculated cob value is nan.")
                    return None

                """ Uncomment to debug"""
                # loss = (ti * (w0 / t0) - w0_hat).dot(ti * (w0 / t0) - w0_hat) + \
                #        (torch.pow(ti, -1) * (w1 * t2) - w1_hat).dot(torch.pow(ti, -1) * (w1 * t2) - w1_hat)
                # if np.mod(st, 400) == -1:
                #    print("step ", step, "loss : ", loss)

                # grads.append(grad.detach().item())

            """ Uncomment to debug the gradient descent. """
            # plt.figure()
            # plt.title("{}".format(i))
            # plt.plot(grads)
            # plt.show()

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
        super().apply_cob(prev_cob, next_cob)
        self.prev_cob = prev_cob

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        super().teleport(prev_cob, next_cob)

    def _get_cob_weight_factor(self, prev_cob: np.ndarray, next_cob: np.ndarray) -> np.ndarray:
        return next_cob

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.base_layer().forward(self, input / self.prev_cob)

    def get_nb_params(self) -> int:
        """Get the number of parameters in the layer (weight and bias).

        Returns:
            number of parameters in the layer.
        """
        nb_params = np.prod(self.weight.shape)
        if self.bias is not None:
            nb_params += np.prod(self.bias.shape)

        nb_params += np.prod(self.running_mean.shape)
        nb_params += np.prod(self.running_var.shape)

        return int(nb_params)

    def get_weights(self, flatten=True, bias=True, get_proxy: bool = True) -> Tuple[torch.Tensor, ...]:
        """Get the weights from the layer.

        Returns:
            tuple of weight tensors.
        """

        # # Check if the weights were updated during training on loading weights.
        # if get_proxy:
        #     self._set_proxy_weights()
        #     if self.bias is not None and bias:
        #         if flatten:
        #             return self.w.flatten(), self.b.flatten()
        #         else:
        #             return self.w, self.b
        #     else:
        #         if flatten:
        #             return self.w.flatten(),
        #         else:
        #             return self.w,
        # else:
        if self.bias is not None and bias:
            if flatten:
                return self.weight.flatten(), self.bias.flatten(), self.running_mean.flatten(), self.running_var.flatten()
            else:
                return self.weight, self.bias
        else:
            if flatten:
                return self.weight.flatten(), self.running_mean.flatten(), self.running_var.flatten()
            else:
                return self.weight,

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
            counter += b_nb_params

        m_shape = self.running_mean.shape
        m_nb_params = np.prod(m_shape)
        m = weights[counter:counter + m_nb_params].reshape(m_shape).type_as(self.running_mean).detach()
        self.register_buffer('running_mean', m)
        counter += m_nb_params


        v_shape = self.running_var.shape
        v_nb_params = np.prod(v_shape)
        v = weights[counter:counter + v_nb_params].reshape(v_shape).type_as(self.running_var).detach()
        self.register_buffer('running_var', v)


class BatchNorm2dCOB(BatchNormMixin, nn.BatchNorm2d):
    reshape_cob = True


class BatchNorm1dCOB(BatchNormMixin, nn.BatchNorm1d):
    reshape_cob = True
