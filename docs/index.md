# lhs

Latin Hypercube Sampling (LHS) is a method for generating a set of points in a multi-dimensional space such that each point is equally spaced in each dimension. This is useful for sampling a function over a multi-dimensional space in a way that is more efficient than a grid-based approach.

The standard LHS algorithm does not take into account sampled points prior to the current iteration and also does not sample from a pool but rather generates a new set of points based on the maximum and minimum values of each dimension.

The `sample` method in this package, however is an adjusted method that takes into consideration both `reference_condition` that are already known samples, and `conditions` which is the condition pool to sample from. 


