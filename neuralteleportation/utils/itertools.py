import itertools
from typing import Dict, List, Mapping, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def dict_values_product(mapping_matrix: Mapping[K, List[V]]) -> List[Dict[K, V]]:
    """Unrolls a mapping defining a list of values for some keys into a list of mapping between keys and single values,
    representing the cartesian product between the values of each of the keys.

    Args:
        mapping_matrix: mapping between keys and lists of values.

    Returns:
        list of mapping between keys and single values.

    Example:
        >>> mapping_matrix = {'a': [1, 2], 'b': [3, 4]}
        >>> dict_values_product(mapping_matrix)
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    # Extended debugging version
    # exploded_matrix = []
    # for matrix_elem_values in itertools.product(*dict_matrix.values()):
    #     exploded_matrix.append(dict(zip(dict_matrix.keys(), matrix_elem_values)))
    # return exploded_matrix

    return [dict(zip(mapping_matrix.keys(), matrix_elem_values))
            for matrix_elem_values in itertools.product(*mapping_matrix.values())]
