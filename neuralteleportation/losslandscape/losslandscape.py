from typing import Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from sklearn.decomposition import PCA
from torch.utils.data.dataset import Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.training.experiment_setup import get_optimizer_from_model_and_config
from neuralteleportation.training.training import test, train_epoch


@dataclass
class LandscapeConfig(TrainingConfig):
    teleport_at: List[int] = 0
    cob_range: float = 0.5
    cob_sampling: str = 'intra_landscape'


linterp_checkpoint_file = '/tmp/linterp_save_checkpoint.pth'
contour_checkpoint_file = '/tmp/contour_save_checkpoint.pth'


def generate_random_2d_vector(weights, ignore_bias_bn: bool = False,
                              normalize: bool = True, seed: int = None) -> List[torch.Tensor]:
    """
        Generates a random vector of size equals to the weights of the model.
    """
    if seed:
        torch.manual_seed(seed)
    if ignore_bias_bn:
        direction = [torch.randn(w.size()) for w in weights]
    else:
        direction = torch.randn(weights.size())
    if normalize:
        normalize_direction(direction, weights, ignore_bias_bn)
    return direction


def generate_direction_vector(checkpoints: List[torch.Tensor], teleport_at: List[int]) -> List[torch.Tensor]:
    """
        Generate the directions vector from model teleportations.

        returns:
            a list containing every teleportation direction.
    """
    res = []
    for n, i in enumerate(teleport_at):
        w_o = checkpoints[i + n]
        w_t = checkpoints[i + (n + 1)]
        res.append(torch.abs(w_o - w_t))

    return res


def normalize_direction(direction: Union[List[torch.Tensor], torch.Tensor],
                        weights: Union[List[torch.Tensor], torch.Tensor],
                        ignore_bias_bn: bool = False):
    """
        Apply a filter normalization to the direction vector.
        d <- d/||d|| * ||w||

         This was use by:
            Authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
            Title: Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.
            Source Code: https://github.com/tomgoldstein/loss-landscape

        This process is a inplace operation.
    """
    assert len(direction) == len(weights), "Direction must have the same size as model weights"
    if ignore_bias_bn:
        for d, w in zip(direction, weights):
            if ignore_bias_bn and d.dim() <= 1:
                d.fill_(0)
            else:
                d.mul_(w.norm() / (d.norm() + 1e-10))
    else:
        direction.mul_(weights.norm() / direction.norm() + 1e-10)


def compute_angle(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
        Calculate the angle in degree between two torch tensors.
    """
    # For some reasons torch.rad2deg and torch.deg2rad function are not there anymore...
    return torch.acos(torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm()))


def generate_teleportation_training_weights(model: NeuralTeleportationModel,
                                            trainset: Dataset,
                                            metric: TrainingMetrics,
                                            config: LandscapeConfig) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
        This will generate a list of weights at a given epoch while training the passed model.
        If teleport_every is different than 0, the model will teleport every time.
    """
    w = [model.get_weights().clone().detach().cpu()]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, drop_last=True)
    optim = get_optimizer_from_model_and_config(model, config)

    for e in range(config.epochs):
        if e in config.teleport_at and config.teleport_at != 0:
            print("Teleporting Model...")
            model.random_teleport(cob_range=config.cob_range, sampling_type=config.cob_sampling)
            w.append(model.get_weights().clone().detach().cpu())
            optim = get_optimizer_from_model_and_config(model, config)

        train_epoch(model, metrics=metric, config=config,
                    train_loader=trainloader, optimizer=optim, epoch=e, device=config.device)
        w.append(model.get_weights().clone().detach().cpu())

    final = w[-1::][0]
    return w, final


def generate_1D_linear_interp(model: NeuralTeleportationModel, param_o: Tuple[torch.Tensor, torch.Tensor],
                              param_t: Tuple[torch.Tensor, torch.Tensor], a: torch.Tensor,
                              trainset: Dataset, valset: Dataset,
                              metric: TrainingMetrics, config: TrainingConfig,
                              checkpoint: dict = None) -> Tuple[list, list, list, list]:
    """
        This is 1-Dimensional Linear Interpolation
        θ(α) = (1−α)θ + αθ′
    """
    loss = []
    loss_v = []
    acc_t = []
    acc_v = []
    w_o, cob_o = param_o
    w_t, cob_t = param_t
    start_at = checkpoint["step"] if checkpoint else 0
    try:
        for step, coord in enumerate(a, start_at):
            # Interpolate the weight from W to T(W),
            # then interpolate the cob for the activation
            # and batchNorm layers only.
            print("step {} of {} - alpha={}".format(step + 1, len(a), coord))
            w = (1 - coord) * w_o + coord * w_t
            cob = (1 - coord) * cob_o + coord * cob_t
            model.set_params(w, cob)
            res = test(model, trainset, metric, config)
            loss.append(res['loss'])
            acc_t.append(res['accuracy'])
            res = test(model, valset, metric, config)
            acc_v.append(res['accuracy'])
            loss_v.append(res['loss'])
    except:
        if not checkpoint:
            checkpoint = {
                'step': step,
                'alpha': a,
                'original_model': param_o,
                'teleported_model': param_t,
                'losses': loss,
                'acc_t': acc_t,
                'acc_v': acc_v,
            }
        else:
            checkpoint['step'] = step
            checkpoint['losses'] = checkpoint['losses'].append(loss)
            checkpoint['acc_t'] = checkpoint['acc_t'].append(acc_t)
            checkpoint['acc_v'] = checkpoint['acc_v'].append(loss)
        torch.save(checkpoint, linterp_checkpoint_file)
        print("A checkpoint was made on step {} of {}".format(step, len(a)))
        # This is to notify the upper level of try/except
        # Since there is no way to know if this is from before teleportation or after teleportation.
        raise

    return loss, acc_t, loss_v, acc_v


def generate_contour_loss_values(model: NeuralTeleportationModel, directions: Tuple[torch.Tensor, torch.Tensor],
                                 weights: torch.Tensor, surface: torch.Tensor, trainset: Dataset,
                                 metric: TrainingMetrics, config: TrainingConfig,
                                 checkpoint: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
        Generate a tensor containing the loss values from a given model.
    """
    loss = []
    acc = []
    delta, eta = directions
    start_at = 0
    if checkpoint:
        start_at = checkpoint['step']
    try:
        for step, (x, y) in enumerate(surface, start_at):
            print("Evaluating step {}: [{:.3f}, {:.3f}]".format(step, x, y))
            x, y = x.to(config.device), y.to(config.device)

            # L (w + alpha*delta + beta*eta)
            changes = (delta * x + eta * y).to(config.device)
            w = torch.add(weights, changes)
            model.set_weights(w)
            results = test(model, trainset, metric, config)

            loss.append(results['loss'])
            acc.append(results['accuracy'])
    except:
        # The reason is that, no matter what, make a checkpoint of the current surface generation.
        if not checkpoint:
            checkpoint = {'step': step,
                          'surface': surface,
                          'loss': loss}
        else:
            checkpoint['step'] = step
            [checkpoint['loss'].append(l) for l in loss]
        torch.save(checkpoint, contour_checkpoint_file)
        print("A checkpoint was made at coord {} of {}".format(x, y))

        # This is to notify the upper level of try/except
        # Since there is no way to know if this is from before teleportation or after teleportation.
        raise

    return np.array(loss), np.array(acc)


def generate_weights_direction(origin_weight, M: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Generate a tensor of the 2 most explanatory directions from the matrix
        M = [W0 - Wn, ... ,Wn-1 - Wn]

        returns:
            Tuple containing the x and y directions vectors. (only use x if doing a 1D plotting)
    """
    M = [m.numpy() for m in M]
    print("Appliying PCA on matrix...")
    pca = PCA(n_components=2)
    pca.fit(M)
    pc1 = torch.tensor(pca.components_[0], dtype=origin_weight.dtype)
    pc2 = torch.tensor(pca.components_[1], dtype=origin_weight.dtype)

    angle = compute_angle(pc1, pc2)

    print("Angle between pc1 and pc2 is {:.3f}".format(angle))
    assert torch.isclose(angle, torch.tensor(1.0, dtype=torch.float), atol=1), "The PCA component " \
                                                                               "are not independent "
    return torch.mul(pc1, origin_weight.cpu()), torch.mul(pc2, origin_weight.cpu())


def generate_weight_trajectory(checkpoints: List[torch.Tensor],
                               direction: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Draw the model weight trajectory over all the train epochs. It is always using a orthogonal projection
        to the 2D space.
    """
    x_direction = direction[0]
    y_direction = direction[1]
    x = []
    y = []
    for w in checkpoints:
        x.append(torch.dot(w, x_direction) / x_direction.norm())
        y.append(torch.dot(w, y_direction) / y_direction.norm())
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y


def plot_contours(x: torch.Tensor, y: torch.Tensor, loss: np.ndarray,
                  weight_traj: Tuple[torch.Tensor, torch.Tensor] = None,
                  teleport_idx: Union[int, List[int]] = None,
                  vmin: int = 0.0, vmax: int = 10, levels: int = 0.5):
    loss = loss.reshape((len(x), len(y)))
    fig = plt.figure()
    plt.contourf(x, y, loss, cmap='coolwarm', levels=np.arange(vmin, vmax, levels))
    plt.colorbar()
    cs = plt.contour(x, y, loss, colors='black', levels=np.arange(vmin, vmax, levels))
    plt.clabel(cs, cs.levels)

    if weight_traj:
        # Plot all the weight points and highlight the teleported one.
        plt.plot(weight_traj[0], weight_traj[1], '-o', c='black')

        if teleport_idx is not None:
            plt.plot(weight_traj[0][teleport_idx], weight_traj[1][teleport_idx], 'x', c='yellow')

    plt.legend()
    plt.tight_layout()
    plt.savefig("contour_{}.png".format(fig.number), format='png')


def plot_interp(loss: List[torch.Tensor], acc_train: List[torch.Tensor], a: torch.Tensor, model1_label="A",
                model2_label="B",
                acc_val: List[torch.Tensor] = None, loss_val: List[torch.Tensor] = None, title=None, savepath=None):
    # Find the nearest value of a=0 and a=1
    idx_o = torch.abs(a - 0).argmin().item()
    idx_t = torch.abs(a - 1).argmin().item()

    loss_color = 'b'
    accuracy_color = 'r'

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if title:
        ax1.set_title(title)

    ax1.set_ylabel("Loss", color=loss_color)
    ax1.plot(a, loss, c=loss_color)  # , 'bo', markersize=1)
    ax1.set_ylim(0, 15)

    ax2.set_ylabel('Accuracy', color=accuracy_color)
    ax2.plot(a, acc_train, c='r')  # , 'ro', markersize=1)

    if acc_val:
        ax2.plot(a, acc_val, '--', c=accuracy_color)  # , 'go', markersize=1)

    if loss_val:
        ax1.plot(a, loss_val, '--', c=loss_color)
    if savepath:
        plt.savefig(savepath, format='png')

    # plt.legend()
    print('Figure has been saved in directory Interpolation. CLOSE figure to continue with the experiment.')
    plt.show()
