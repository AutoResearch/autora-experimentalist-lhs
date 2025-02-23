"""
Example Experimentalist
"""
import itertools
import math
import random
from functools import cmp_to_key
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from autora.variable import VariableCollection


def pool(
    variables: VariableCollection,
    num_samples: int = 1,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    The Latin Hypercube Pooler generates a pool of experimental conditions
    using maximum and minumum values of each variable.

    Args:
        variables: The variable definitions
        num_samples: The number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        A DataFrame with the experimental conditions

    Example:
        >>> from autora.variable import Variable, VariableCollection
        >>> _ivs = [
        ...     Variable("a", value_range=(0, 1)),
        ...     Variable("b", value_range=(-2, -1)),
        ...     Variable("c", value_range=(3, 4))]
        >>> _variables = VariableCollection(independent_variables=_ivs)
        >>> pool(_variables, 8, random_state=42)
                  a         b         c
        0  0.319713 -1.987495  3.137515
        1  0.111605 -1.631764  3.838350
        2  0.446090 -1.456531  3.210961
        3  0.014899 -1.390681  3.752678
        4  0.513268 -1.900581  3.324942
        5  0.772471 -1.889780  3.794633
        6  0.904715 -1.496751  3.402910
        7  0.849070 -1.329875  3.577740
    """
    sampler = random.Random(random_state)

    ivs = [v.name for v in variables.independent_variables]
    for var in variables.independent_variables:
        if var.value_range is None:
            raise ValueError("Variable value range must be defined")
        if var.allowed_values is not None:
            raise Warning("Variable allowed values are ignored in the lhs pooler")

    num_hypercubes = num_samples

    num_bins_per_variable = math.ceil(num_hypercubes ** (1 / len(ivs)))

    tmp = {}
    for var in variables.independent_variables:
        _max = var.value_range[1]
        _min = var.value_range[0]
        d = (_max - _min) / num_bins_per_variable
        tmp[var.name] = [
            [_min + i * d, _min + (i + 1) * d] for i in range(num_bins_per_variable)
        ]

    # Create the hypercubes with all possible combinations
    hypercubes = [
        dict(zip(tmp.keys(), values)) for values in itertools.product(*tmp.values())
    ]

    sampled_hypercubes = []
    remaining_samples = num_samples
    # Sample from the hypercubes
    while remaining_samples > 0:
        if remaining_samples >= len(hypercubes):
            sampled_hypercubes.extend(hypercubes)
            remaining_samples -= len(hypercubes)
        else:
            sampled_hypercubes.extend(sampler.sample(hypercubes, remaining_samples))
            break

    res = {}

    for sample in sampled_hypercubes:
        for key in sample:
            if key not in res:
                res[key] = [
                    sampler.random() * (sample[key][1] - sample[key][0])
                    + sample[key][0]
                ]
            else:
                res[key].append(
                    sampler.random() * (sample[key][1] - sample[key][0])
                    + sample[key][0]
                )

    return pd.DataFrame(res)


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    num_samples: int = 1,
    less_then: Optional[Dict[str, Callable[[Any, Any], bool]]] = None,
    random_state: Optional[int] = None,
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
        less_then: For each column in the conditions, an order function can be defined to
            use for stratification
        random_state: Random seed for reproducibility
    Returns:
        Sampled pool of experimental conditions

    Example:
        >>> import random
        >>> cnd = pd.DataFrame({
        ...    'a': [1, 2, 3, 4, 5],
        ...    'b': [1, 2, 3, 4, 5]}
        ... )
        >>> ref_cnd = pd.DataFrame({
        ...    'a': [1, 2],
        ...    'b': [1, 2]}
        ... )
        >>> n = 2
        >>> sample(cnd, ref_cnd, n, random_state=42)
           a  b
        0  5  5
        1  4  4

    """

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
        if less_then is not None and col in less_then:
            _less_then = less_then[col]

            def compare(x, y):
                if _less_then(x, y):
                    return -1
                elif _less_then(y, x):
                    return 1
                return 0

            bins[col] = sorted(
                _conditions[col],
                key=cmp_to_key(compare),
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
        _sum = 0
        if not skip:
            for ref in reference_condition_dict:
                if _elem_in_hypercube(ref, hc, less_then):
                    _sum += 1
                    refs += 1
        hypercube_counts.append({"hypercube": hc, "samples": _sum})
        if refs >= len(reference_condition_dict):
            skip = True

    # Sample from the hypercubes
    n_to_sample = num_samples
    samples = []
    idx = 0
    sampler = random.Random(random_state)
    while n_to_sample > 0:

        _hypercubes = [hc for hc in hypercube_counts if hc["samples"] == idx]

        if len(_hypercubes) <= n_to_sample:
            samples.extend(_hypercubes)
            n_to_sample -= len(_hypercubes)
            for hc in hypercube_counts:
                if hc["samples"] == idx:
                    if not isinstance(hc["samples"], (int, float)):
                        raise TypeError(
                            f"Unexpected type for hc['samples']: {type(hc['samples'])}"
                        )
                    hc["samples"] += 1
            idx += 1

        else:
            samples.extend(sampler.sample(_hypercubes, n_to_sample))
            break

    res = {}
    # sample a condition from each hypercube while ensuring that the condition is
    # included in the conditions
    for hypercube in samples:
        _sample = _sample_condition_from_hypercube(
            hypercube["hypercube"], condition_dicts, sampler
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
    less_then: Optional[Dict[str, Callable[[Any, Any], bool]]] = None,
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
        if not less_then or key not in less_then:
            if not _min <= value <= _max:
                return False
        else:
            if not less_then[key](_min, value) or not less_then[key](value, _max):
                return False
    return True  # No match found


def _sample_condition_from_hypercube(hypercube, conditions, sampler=None):
    """
    Samples a valid condition from each hypercube, ensuring it exists in the allowed conditions.

    Args:
        hypercube (list): List of hypercubes, where each hypercube is a dict with
        'hypercube' and 'samples' keys.
        conditions (list): List of allowed conditions
            (each as a dictionary {a: value, b: value, c: value}).

    Returns:
        list: A list of sampled conditions, one for each hypercube.
    """
    if sampler is None:
        sampler = random.Random()
    _conditions = conditions.copy()
    _sample = sampler.choice(_conditions)

    while not _elem_in_hypercube(_sample, hypercube):
        _conditions.remove(_sample)
        _sample = sampler.choice(_conditions)

    return _sample
