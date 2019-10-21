import numpy as np
import pandas as pd
from numpy.linalg import norm


def init(A: np.ndarray):
    Dr = A.sum(axis = 1)
    Dc = A.sum(axis = 0)
    
    Dr1_A = np.diag(1/Dr).dot(A)
    Dc1_A = np.diag(1/Dc).dot(A.T)
    
    return Dr1_A, Dc1_A


def rand_unit(n: int):
    x = np.random.rand(n)
    x = np.abs(x)
    x = x / norm(x)
    return x


def r1svd(A: np.ndarray, threshold: float = 1E-4):
    Dr1_A, Dc1_A = init(A)
    
    n, m = A.shape
    
    u = rand_unit(n)
    v = rand_unit(m)
    
    u_prev = u
    v_prev = v
    
    gamma_prev = -np.inf
    
    it = 0
    
    while True:
        it += 1
        
        v = np.dot(Dc1_A, u_prev)
        v = v/norm(v)
        
        u = np.dot(Dr1_A, v_prev)
        u = u/norm(u)
        
        gamma = norm(u - u_prev) + norm(v - v_prev)
                
        if abs(gamma - gamma_prev) <= threshold:
            print(f'Converged in {it} iterations.\n u = {u}\n v = {v}')
            return u,v
        
        gamma_prev = gamma
        u_prev = u
        v_prev = v


def reorder(M: np.ndarray, by: np.ndarray):
    idx = np.argsort(by)
    return M[np.ix_(idx)]