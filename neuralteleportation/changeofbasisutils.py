import numpy as np
from torch import Tensor, tensor


available_sampling_types = ["intra_landscape", "inter_landscape", "positive", "negative", "centered"]


def get_available_cob_sampling_types():
    return available_sampling_types


def get_random_cob(range_cob: float, size: int, sampling_type: str = 'intra_landscape', center=1,
                   requires_grad: bool=False) -> Tensor:
    """
        Return random change of basis:

        'intra_landscape' - in interval [1 - range_cob, 1 + range_cob]
        'inter_landscape' - equally in intervals [-1 - range_cob, -1 + range_cob] and [1 - range_cob, 1 + range_cob]
        'positive'         - in interval [0, range_cob]
        'negative'         - in interval [-range_cob, 0]
        'centered'         - in interval [center - range_cob, center + range_cob]

    Args:
        range_cob (float):    range_cob for the change of basis.
        size (int):         size of the returned array.
        sampling_type (str):      label for type of sampling for change of basis
        center:             The center of the uniform distribution with which to sample in case of
                            sampling_type=centered
        requires_grad (bool): whether the cob tensor should require gradients
    Returns:
        torch.Tensor of size size.
    """
    # Change of basis in interval [1-range_cob, 1+range_cob]
    if sampling_type == 'intra_landscape':
        assert not (range_cob > center or center <= 0), 'This range for change of basis sampling allows for negative ' \
                                                        'changes of basis.'
        if center != 1:
            print('Warning: The change of basis sampling is not centered at 1. But no negative change of basis'
                  ' will be produced.')
        cob = np.random.uniform(low=-range_cob, high=range_cob, size=size).astype(np.float) + 1

    # Change of basis equally in intervals [-1-range_cob, -1+range_cob] and [1-range_cob, 1+range_cob]
    elif sampling_type == 'inter_landscape':
        samples = np.random.randint(0, 2, size=size)
        cob = np.zeros_like(samples, dtype=np.float)
        cob[samples == 1] = np.random.uniform(low=-1-range_cob, high=-1+range_cob, size=samples.sum())
        cob[samples == 0] = np.random.uniform(low=1-range_cob, high=1+range_cob, size=(len(samples) - samples.sum()))

    # Change of basis in interval [center- range_cob, center + range_cob]
    elif sampling_type == 'centered':
        assert not (range_cob > center or center <= 0), 'This range for change of basis sampling allows for negative ' \
                                                        'changes of basis.'
        cob = np.random.uniform(low=center - range_cob, high=center + range_cob, size=size).astype(np.float)

    # Change of basis in interval [0, range_cob]
    elif sampling_type == 'positive':
        cob = np.random.uniform(low=0, high=range_cob, size=size).astype(np.float)

    # Change of basis in interval [-range_cob, 0]
    elif sampling_type == 'negative':
        cob = np.random.uniform(low=-range_cob, high=0, size=size).astype(np.float)

    else:
        raise ValueError("Invalid sampling type. Sampling types allowed: "
                         "intra_landscape, inter_landscape, positive, negative, centered")

    return tensor(cob, requires_grad=requires_grad)


if __name__ == '__main__':
    cob = get_random_cob(10, 10, requires_grad=True)
    print(cob)
