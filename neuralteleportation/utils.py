import numpy as np


def get_random_positive_cob(range: int, size: int) -> np.ndarray:
    """
        Return random change of basis between 1/range and range.
        Half of samples are in [1/range, 1[ and other half is in ]1, range]
    Args:
        range (int): range for the change of basis.
        size (int): size of the returned array.

    Returns:
        ndarray of size size.
    """
    samples = np.random.randint(0, 2, size=size)
    cob = np.zeros_like(samples, dtype=np.float)
    cob[samples == 1] = np.random.uniform(low=1 / range, high=1, size=samples.sum())
    cob[samples == 0] = np.random.uniform(low=1, high=range, size=(len(samples) - samples.sum()))

    return cob

def get_random_cob(range: int, size: int) -> np.ndarray:
    """
        Return random change of basis between -range and range.

    Args:
        range (int): range for the change of basis.
        size (int): size of the returned array.

    Returns:
        ndarray of size size.
    """

    return np.random.uniform(low=-range, high=range, size=size).astype(np.float)


