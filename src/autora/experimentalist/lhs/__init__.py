"""
Example Experimentalist
"""
import itertools
import random
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd

from autora.variable import VariableCollection


def pool(variables: VariableCollection, num_samples: int = 1) -> pd.DataFrame:
    raise NotImplementedError


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    num_samples: int = 1,
    less_then: Dict[str, Callable[[any, any], bool]] = None,
) -> pd.DataFrame:
    """
    The Latin Hypercube Sampler samples from a pool of experimental conditions using
    stratification with bin ordering based on a distance metric or "<=" constraint.

    First the conditions are sorted into num_samples + num_ref hypercubes. Then hypercubes
    that already contain reference conditions are removed. From each remaining hypercube
    a random sample is selected.

    Args:
        conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        reference_conditions: The sampler might use reference conditons
        num_samples: number of experimental conditions to select
        less_then: For each column in the conditions, an order function can be defined to use for stratification
    Returns:
        Sampled pool of experimental conditions

    Example:
        >>> import random
        >>> random.seed(42)
        >>> conditions = pd.DataFrame({
        ...    'a': [1, 2, 3, 4, 5],
        ...    'b': [1, 2, 3, 4, 5]}
        ... )
        >>> reference_conditions = pd.DataFrame({
        ...    'a': [1, 2],
        ...    'b': [1, 2]}
        ... )
        >>> num_samples = 2
        >>> _sample(conditions, reference_conditions, num_samples)
           a  b
        0  5  5
        1  4  4

    """

    import math
    from functools import cmp_to_key

    # Ensure we don't modify the original DataFrames
    _conditions = conditions.copy()
    _reference_conditions = reference_conditions.copy()

    # Calculate the number of hypercubes
    num_reference = len(_reference_conditions)
    num_hypercubes = num_samples + num_reference

    # Calculate the number of bins in each dimension
    num_bins_per_condition = math.ceil(num_hypercubes ** (1 / len(_conditions.columns)))
    len_bins = math.ceil(len(_conditions) / num_bins_per_condition)

    # Create sorted lists
    bins = {}
    for col in _conditions.columns:
        if less_then and col in less_then:
            bins[col] = sorted(
                _conditions[col],
                key=cmp_to_key(
                    lambda x, y: -1
                    if less_then[col](x, y)
                    else (1 if less_then[col](y, x) else 0)
                ),
            )
        else:
            bins[col] = sorted(_conditions[col])

    # Create the intervals for each column
    for key in bins:
        bins[key] = [
            bins[key][i : i + len_bins] for i in range(0, len(bins[key]), len_bins)
        ]

    # Create the hypercubes with all possible combinations
    _hypercubes = [
        dict(zip(bins.keys(), values)) for values in itertools.product(*bins.values())
    ]

    # Filter out hypercubes that contain no conditions
    condition_dicts = [row.to_dict() for i, row in _conditions.iterrows()]

    reference_condition_dict = [
        row.to_dict() for i, row in _reference_conditions.iterrows()
    ]

    hypercubes = []
    for hc in _hypercubes:
        for cond in condition_dicts:
            if _elem_in_hypercube(cond, hc, less_then):
                hypercubes.append(hc)
                break
    hypercube_counts = []
    refs = 0
    skip = False
    for hc in hypercubes:
        sum = 0
        if not skip:
            for ref in reference_condition_dict:
                if _elem_in_hypercube(ref, hc, less_then):
                    sum += 1
                    refs += 1
        hypercube_counts.append({"hypercube": hc, "samples": sum})
        if refs >= len(reference_condition_dict):
            skip = True

    # Sample from the hypercubes
    n_to_sample = num_samples
    samples = []
    idx = 0
    while n_to_sample > 0:

        _hypercubes = [hc for hc in hypercube_counts if hc["samples"] == idx]

        if len(_hypercubes) <= n_to_sample:
            samples.extend(_hypercubes)
            n_to_sample -= len(_hypercubes)
            for hc in hypercube_counts:
                if hc["samples"] == idx:
                    hc["samples"] += 1
            idx += 1

        else:
            samples.extend(random.sample(_hypercubes, n_to_sample))
            break

    res = {}
    # sample a condition from each hypercube while ensuring that the condition is included in the conditions
    print("sampling conditions...")
    for hypercube in samples:
        _sample = _sample_condition_from_hypercube(
            hypercube["hypercube"], condition_dicts
        )
        for key in _sample:
            if key not in res:
                res[key] = [_sample[key]]
            else:
                res[key].append(_sample[key])

    return pd.DataFrame(res)


def sample_experiment_data(
    conditions: Union[pd.DataFrame, np.ndarray],
    experiment_data: Union[pd.DataFrame, np.ndarray],
    num_samples: int = 1,
    less_then: Dict[str, Callable[[any, any], bool]] = None,
) -> pd.DataFrame:
    raise NotImplementedError


def _elem_in_hypercube(el, hypercube, less_then=None):
    """
    Example:
        >>> _elem_in_hypercube({'a': 1, 'b': 2}, {'a': [1, 2, 3], 'b': [2, 3, 4]})
        True

        >>> _elem_in_hypercube({'a': 1, 'b': 2}, {'a': [1, 2, 3], 'b': [3, 4, 5]})
        False

        >>> _elem_in_hypercube({'a': 1, 'b': 2}, {'a': [1, 2, 3], 'b': [1, 2, 3]})
        True
    """
    # return tuple(el[k] for k in hypercube) in {tuple(map(tuple, zip(*hypercube.values())))}

    # Ensure all keys exist in the hypercube
    if not all(key in hypercube for key in el):
        return False

    # Check if the combination exists at the same index
    for key, value in el.items():
        _min = hypercube[key][0]
        _max = hypercube[key][-1]
        if not less_then or not key in less_then:
            if not _min <= value <= _max:
                return False
        else:
            if not less_then[key](_min, value) or not less_then[key](value, _max):
                return False
    return True  # No match found


def _sample_condition_from_hypercube(hypercube, conditions):
    """
    Samples a valid condition from each hypercube, ensuring it exists in the allowed conditions.

    Args:
        hypercube (list): List of hypercubes, where each hypercube is a dict with 'hypercube' and 'samples' keys.
        conditions (list): List of allowed conditions (each as a dictionary {a: value, b: value, c: value}).

    Returns:
        list: A list of sampled conditions, one for each hypercube.
    """
    _conditions = conditions.copy()
    _sample = random.choice(_conditions)

    while not _elem_in_hypercube(_sample, hypercube):
        _conditions.remove(_sample)
        _sample = random.choice(_conditions)

    return _sample
