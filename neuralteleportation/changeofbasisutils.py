import numpy as np

def get_random_cob(range: int, size: int, SamplingType='usual') -> np.ndarray:
    """
        Return random change of basis between -range+1 and range+1.
        'usual' - in interval [1-range,1+range]
        'symmetric' - equally in intervals [-1-range,-1+range] and [1-range,1+range]
        'negative' - in interval [-1-range,-1+range]
        'zero' - in interval [-range,range]

    Args:
        range (int): range for the change of basis. Recommended between 0 and 1, but can take any positive range. 
        size (int): size of the returned array.
	SamplingType: label for type of sampling for change of basis
    Returns:
        ndarray of size size.
    """
    # Change of basis in interval [1-range,1+range]
    if SamplingType == 'usual':
    	return np.random.uniform(low=-range, high=range, size=size).astype(np.float) + 1
    	
    # Change of basis in intervals [-1-range,-1+range] and [1-range,1+range]
    elif SamplingType == 'symmetric':
	    samples = np.random.randint(0, 2, size=size)
	    cob = np.zeros_like(samples, dtype=np.float)
	    cob[samples == 1] = np.random.uniform(low=-1-range, high=-1+range, size=samples.sum())
	    cob[samples == 0] = np.random.uniform(low=1-range, high=1+range, size=(len(samples) - samples.sum()))
	    return cob
    	
    # Change of basis in interval [-1-range,-1+range]
    elif SamplingType == 'negative':
    	return np.random.uniform(low=-range, high=range, size=size).astype(np.float) - 1
    	
    # Change of basis in interval [-range,range]
    # This will produce very big weights in the network. Use only if needed.
    elif SamplingType == 'zero':
    	return np.random.uniform(low=-range, high=range, size=size).astype(np.float)
