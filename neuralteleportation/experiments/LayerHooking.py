import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from neuralteleportation.layerhook import LayerHook
from neuralteleportation.metrics import accuracy
from neuralteleportation.training import experiment_setup
from neuralteleportation.training.config import *
from neuralteleportation.training.training import train

__models__ = experiment_setup.get_model_names()


def argument_parser():
    """
        Simple argument parser for the experiement.

        Ex: python layerhooking.py --batch_size 10 --epochs 5 --model resenet --layer_name layer1

    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the layer hook experiment.')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model", type=str, default="resnet18COB", choices=__models__)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--layer_name", type=str, default="conv1",
                        help="the name of the layer that is going to be hooked. user must go verify the right naming "
                             "format for the model they are trying to hook")
    parser.add_argument("--num_teleportation", type=int, default=2)
    parser.add_argument("--cob_range", type=float, default=10.0)
    parser.add_argument("--cob_sampling", type=str, default="intra_landscape")
    parser.add_argument("--show_original", action="store_true", default=True,
                        help="enable the plotting of the original image.")
    return parser.parse_args()


if __name__ == "__main__":
    # Hook Callback example, it can be anything you want.
    # The Hook Callback always need the signature (self, input, output).
    def hookCallback(self, input, output):
        """
            small callback that prints the first outputed image to a pyplot figure.
        """
        np_out = output.detach().cpu().numpy()
        plt.figure()
        plt.imshow(np_out[0, 0, :, :])
        plt.colorbar()
        plt.clim(vmin=-100, vmax=256)


    args = argument_parser()
    batch_size = args.batch_size

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = transforms.ToTensor()
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
    trainset, valset, testset = experiment_setup.get_dataset_subsets(args.dataset, transform=transform)

    data_train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    w, h = trainset.data.shape[1:3]
    if trainset.data.ndim == 4:
        dims = trainset.data.shape[3]
    else:
        dims = 1

    test_img = torch.as_tensor(testset.data[0], dtype=torch.float32).to(device=device)
    test_img = torch.reshape(test_img, (1, w, h, dims))
    test_img = test_img.permute(0, 3, 1, 2)

    if args.show_original:
        plt.figure()
        plt.imshow(testset.data[0])
        plt.colorbar()
        plt.title("Original Image")

    # Change the model to a teleportable model.
    net = experiment_setup.get_model(args.dataset, args.model, device=device.type)
    metric = TrainingMetrics(torch.nn.CrossEntropyLoss(), [accuracy])
    config = TrainingConfig(epochs=args.epochs, device=device.type, batch_size=batch_size)
    train(net, train_dataset=trainset, metrics=metric, config=config)

    # Attach the hook to a specific layer
    hook_state = (args.layer_name, hookCallback)
    hook = LayerHook(net, hook_state)

    # We do a forward propagation to illustrate the example.
    net.eval()
    img_output = net(test_img)
    plt.title("non-teleported")

    teleported_prediction = []
    for n in range(args.num_teleportation):
        net.random_teleport(cob_range=args.cob_range, sampling_type=args.cob_sampling)
        teleported_img_output = net(test_img)
        plt.title("teleportation number: %s" % (n + 1))
        img_prediction = torch.max(img_output, 1)
        teleported_prediction.append(torch.max(teleported_img_output, 1))

    print("=========Image Prediction=========")
    print(img_prediction)
    for pred in teleported_prediction:
        print(pred)

    plt.show()
