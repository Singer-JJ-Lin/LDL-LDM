import numpy as np
from scipy.spatial import distance

# Projecting V into probability simplex
def project_onto_simplex(V):
    n, m = V.shape
    sorted_V = np.sort(V, axis=0)[::-1]
    p = np.arange(1, n + 1).reshape(-1, 1)
    cumsum = np.cumsum(sorted_V, axis=0) - 1
    return np.maximum(cumsum / p, 0)

def Chebyshev(Y, Y_hat):
    return np.max(np.abs(Y - Y_hat), 1).mean()

def Clark(Y, Y_hat):
    diff_abs = np.abs(Y - Y_hat)
    sum = Y + Y_hat
    diff_abs_square = np.power(diff_abs, 2)
    sum_square = np.power(sum, 2)
    return np.sqrt(np.sum(diff_abs_square / sum_square, axis=1)).mean()

def Canberra(Y, Y_hat):
    diff_abs = np.abs(Y - Y_hat)
    sum = Y + Y_hat
    return np.sum(diff_abs / sum, 1).mean()

def KL_Divergence(Y, Y_hat):
    return np.sum(Y * (np.log(Y) - np.log(Y_hat)), axis=1).mean()

def Cosine(Y, Y_hat):
    U = np.multiply(Y, Y_hat)
    U = np.sum(U, axis=1).reshape(-1, 1)
    D = (np.linalg.norm(Y,axis=1).reshape(-1, 1) * np.linalg.norm(Y_hat,axis=1).reshape(-1, 1))
    return (U/D).mean()

def Intersection(Y, Y_hat):
    return 0.5 * np.sum(np.abs(Y - Y_hat), axis=1).mean()








