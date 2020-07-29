import torch


def accuracy_topk(output, target, topk=1):
    """Computes the accuracy of output with respect to target

    Args:
        output: torch.Tensor, model output [N, C]
        target: torch.Tensor, target [N]
        topk: int, k top predictions to compare

    Returns:
        float, accuracy
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct[:topk].view(-1).float().sum(0)
        acc = correct / batch_size
        return acc.item()


def accuracy(output, target):
    return accuracy_topk(output, target, topk=1)


def accuracy_top5(output, target):
    return accuracy_topk(output, target, topk=5)
