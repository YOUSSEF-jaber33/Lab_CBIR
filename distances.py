import numpy as np
from scipy.spatial import distance
 
def euclidean(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return np.sqrt(np.sum((v1 - v2) ** 2))
 
def manhattan(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return np.sum(np.abs(v1 - v2))
 
def chebyshev(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return np.max(np.abs(v1 - v2))
 
def canberra(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return np.sum(np.abs(v1 - v2) / (np.abs(v1) + np.abs(v2)))