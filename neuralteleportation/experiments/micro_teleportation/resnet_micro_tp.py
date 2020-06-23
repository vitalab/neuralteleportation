import torch.optim as optim

from collections import defaultdict
from models.model_zoo import resnetcob
from utils.micro_tp_utils import *

# ANSI escape code for colored console text
red = "\033[31m"
reset = "\033[0m"


def train(model, criterion, train_dataset, val_dataset=None, optimizer=None, metrics=None, epochs=10, batch_size=32,
          device='cpu'):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        train_step(model, criterion, optimizer, train_loader, epoch, device=device)
        if val_dataset:
            val_res = test(model, criterion, metrics, val_dataset, device=device)
            print("Validation: {}".format(val_res))


def train_step(model, criterion, optimizer, train_loader, epoch, device='cpu'):

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       batch_idx / len(train_loader), loss.item()))


def test(model, criterion, metrics, dataset, batch_size=32, device='cpu'):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            results['loss'].append(criterion(output, target).item())

            if metrics is not None:
                batch_results = compute_metrics(metrics, y=target, y_hat=output, to_tensor=False)
                for k in batch_results.keys():
                    results[k].append(batch_results[k])

    results = pd.DataFrame(results)
    return dict(results.mean())


def compute_metrics(metrics, y_hat, y, prefix='', to_tensor=True):
    results = {}
    for metric in metrics:
        m = metric(y_hat, y)
        if to_tensor:
            m = torch.tensor(m)
        results[prefix + metric.__name__] = m
    return results


if __name__ == '__main__':
    from torchvision.datasets import MNIST, CIFAR10, CIFAR100
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    import torch.nn as nn

    trans = list()
    trans.append(transforms.Resize(size=224))
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    mnist_train = MNIST('/tmp', train=True, download=True, transform=trans)
    mnist_val = MNIST('/tmp', train=False, download=True, transform=trans)
    mnist_test = MNIST('/tmp', train=False, download=True, transform=trans)

    resnet_model = resnetcob.resnet18COB(input_channels=1)

    optim = torch.optim.Adam(params=resnet_model.parameters(), lr=.01)
    metrics = [accuracy]
    loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train(resnet_model, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_val, optimizer=optim, metrics=metrics,
          epochs=1, device=device, batch_size=16)
    print(test(resnet_model, loss, metrics, mnist_test, device=device))

    dot_product(resnet_model, dataset=mnist_test, network_descriptor='ResNet on MNIST')

    trans = list()
    trans.append(transforms.ToTensor())
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    train_set = CIFAR10('/tmp', train=True, download=True,
                        transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                      transforms.RandomCrop(32, 4),
                                                      transforms.ToTensor()]))
    val_set = CIFAR10('/tmp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = CIFAR10('/tmp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    resnet_model = resnetcob.resnet18COB(pretrained=False, input_channels=3)

    optim = torch.optim.Adam(params=resnet_model.parameters(), lr=.01)
    metrics = [accuracy]
    loss = nn.CrossEntropyLoss()

    train(resnet_model, criterion=loss, train_dataset=train_set, val_dataset=val_set, optimizer=optim, metrics=metrics,
          epochs=1, device=device, batch_size=16)
    print(test(resnet_model, loss, metrics, test_set, device=device))

    dot_product(network=resnet_model, dataset=test_set, network_descriptor='VGG on CIFAR10', device=device)

    train_set = CIFAR100('/tmp', train=True, download=True,
                         transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       transforms.RandomCrop(32, 4),
                                                       transforms.ToTensor()]))
    val_set = CIFAR100('/tmp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = CIFAR100('/tmp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    resnet_model = resnetcob.resnet18COB(pretrained=False, input_channels=3)

    train(resnet_model, criterion=loss, train_dataset=train_set, val_dataset=val_set, optimizer=optim, metrics=metrics,
          epochs=1, device=device, batch_size=16)
    print(test(resnet_model, loss, metrics, test_set, device=device))

    dot_product(network=resnet_model, dataset=test_set, network_descriptor='VGG on CIFAR100', device=device)