import torch


def accuracy(output, target):
    """Computes the accuracy of output with respect to target

    Args:
        output: torch.Tensor, model output [N, C]
        target: torch.Tensor, target [N]

    Returns:
        float, accuracy
    """
    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct / batch_size
        return acc
