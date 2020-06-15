"""
Closed form optimisation on a network's change of basis to find the change of basis that teleports
to a given set of weights.
"""
import torch
import torch.nn as nn

from neuralteleportation.metrics import accuracy
from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.training.experiment_setup import get_mnist_datasets
from neuralteleportation.training.training import train, test
import argparse
from os.path import join as pjoin

from neuralteleportation.utils.pathutils import get_nonexistent_path


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights1", type=str, default=None, help='Weights for the first model')
    parser.add_argument("--weights2", type=str, default=None, help='Weights for the second model')
    parser.add_argument("--epochs", type=int, default=0, help='Number of epochs to train the networks')
    parser.add_argument("--save_path", type=str, default='coboptim', help='Path to save weights without extension')
    return parser.parse_args()


if __name__ == '__main__':

    args = argument_parser()
    save_path = get_nonexistent_path(args.save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(epochs=args.epochs, device=device, shuffle_batches=True)
    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    mnist_train, mnist_val, mnist_test = get_mnist_datasets()

    sample_input_shape = (1, 1, 28, 28)

    model1 = MLPCOB(num_classes=10).to(device)
    model1 = NeuralTeleportationModel(network=model1, input_shape=sample_input_shape)
    if args.weights1 is not None:
        model1.load_state_dict(torch.load(args.weights1))
    train(model1, train_dataset=mnist_train, metrics=metrics, config=config, val_dataset=mnist_test)
    torch.save(model1.state_dict(), pjoin(save_path, 'model1.pt'))
    print("Model 1 test results: ", test(model1, mnist_test, metrics, config))

    model2 = MLPCOB(num_classes=10).to(device)
    model2 = NeuralTeleportationModel(network=model2, input_shape=sample_input_shape)
    if args.weights2 is not None:
        model2.load_state_dict(torch.load(args.weights2))
    train(model2, train_dataset=mnist_train, metrics=metrics, config=config, val_dataset=mnist_test)
    torch.save(model2.state_dict(), pjoin(save_path, 'model2.pt'))
    print("Model 2 test results: ", test(model2, mnist_test, metrics, config))

    # Compare the output of the two models for a given input.
    x, y = mnist_train[0]
    pred1 = model1(x.to(device))
    pred2 = model2(x.to(device))
    print("Model 1 prediction: ", pred1)
    print("Model 2 prediction: ", pred2)
    print("Pred diff", (pred1 - pred2).abs())
    print("Pred diff mean", (pred1 - pred2).abs().mean())

    print("Initial: w1 - w2 ([:100]): ", (model1.get_weights() - model2.get_weights()).abs()[:100])
    print("Initial: w1 - w2 ([-100:]): ", (model1.get_weights() - model2.get_weights()).abs()[-100:])

    w1 = model1.get_weights()
    w2 = model2.get_weights()
    diff = (w1.detach().cpu() - w2.detach().cpu()).abs().mean()
    print("Initial weight difference :", diff)

    w1 = model1.get_weights(concat=False, flatten=False, bias=False)
    w2 = model2.get_weights(concat=False, flatten=False, bias=False)
    calculated_cob = model1.calculate_cob(w1, w2, concat=True, eta=0.00001, steps=6000)
    torch.save(calculated_cob, pjoin(save_path, 'calculated_cob.pt'))

    model1.teleport(calculated_cob)

    w1 = model1.get_weights()
    w2 = model2.get_weights()
    diff = (w1.detach().cpu() - w2.detach().cpu()).abs().mean()
    print("Predicted weight difference :", diff)

    w1 = model1.get_weights(concat=False, flatten=False, bias=False)
    w2 = model2.get_weights(concat=False, flatten=False, bias=False)

    print("Weight difference by layer:")
    for i in range(len(w1)):
        print('layer : ', i)
        print("w1  - w2 = ", (w1[i].detach().cpu() - w2[i].detach().cpu()).abs().sum())
        print("w1: ", w1[i].detach().cpu().flatten()[:10])
        print("w2: ", w2[i].detach().cpu().flatten()[:10])
