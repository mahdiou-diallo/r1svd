import numpy as np
import pandas as pd
from numpy.linalg import norm

import matplotlib.pyplot as plt


def _rand_unit(n: int):
    x = np.random.rand(n)
    x = np.abs(x)
    x = x / norm(x)
    return x


def _init_diagonal_matrices(A):
    Dr = A.sum(axis=1)
    Dc = A.sum(axis=0)

    Dr1_A = np.diag(1/Dr) @ A
    Dc1_A = np.diag(1/Dc) @ A.T

    return Dr1_A, Dc1_A


def _reorder(M: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    idx_r = np.argsort(rows)
    idx_c = np.argsort(cols)
    return M[np.ix_(idx_r, idx_c)], rows[idx_r], cols[idx_c]


class RankOneSvd:
    """ """
    def __init__(self, threshold=1E-4):
        self.threshold = threshold

    def fit_transform(self, A: np.ndarray):
        self.A_ = A
        Dr1_A, Dc1_A = _init_diagonal_matrices(A)

        n, m = self.A_.shape

        u = _rand_unit(n)
        v = _rand_unit(m)

        u_prev = u
        v_prev = v

        gamma_prev = -np.inf

        it = 0

        while True:
            it += 1

            v = Dc1_A @ u_prev
            v = v/norm(v)

            u = Dr1_A @ v_prev
            u = u/norm(u)

            gamma = norm(u - u_prev) + norm(v - v_prev)

            if abs(gamma - gamma_prev) <= self.threshold:
                print(f'Converged in {it} iterations.')
                # print('u = {u}\n v = {v}')
                self.u_ = u
                self.v_ = v
                self.A_sorted_, self.u_sorted_, self.v_sorted_ = _reorder(A, u, v)
                return self

            gamma_prev = gamma
            u_prev = u
            v_prev = v
    
    def get_row_labels(self):
        du = np.concatenate([[0], np.diff(self.u_sorted_)])
        m = np.mean(du)
        return np.cumsum(du > m)
    
    def get_col_labels(self):
        dv = np.concatenate([[0], np.diff(self.v_sorted_)])
        m = np.mean(dv)
        return np.cumsum(dv > m)

    def plot_original_matrix(self, matrix='A'):
        if matrix == 'A':
            plt.spy(self.A_)
            return
        if matrix == 'Sr':
            plt.spy(self.A_ @ self.A_.T)
            return
        if matrix == 'Sc':
            plt.spy(self.A_.T @ self.A_)
            return

    def plot_reordered_matrix(self, matrix='A'):
        if matrix == 'A':
            plt.spy(self.A_sorted_)
            return
        if matrix == 'Sr':
            plt.spy(_reorder(self.A_ @ self.A_.T, self.u_, self.u_))
            return
        if matrix == 'Sc':
            plt.spy(_reorder(self.A_.T @ self.A_, self.v_, self.v_))
            return

    def plot_u(self, transpose=False):
        if transpose:
            plt.plot(np.sort(self.u_), np.arange(self.u_.shape[0]))
        else:
            plt.plot(np.sort(self.u_))

    def plot_v(self):
        plt.plot(np.sort(self.v_))


if __name__ == '__main__':
    X = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

    # columns = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
    # index = ["HighSchool", "AgricultCoop", "Railstation", "OneRoomSchool", "Veterinary", "NoDoctor", "NoWaterSupply",  "PoliceStation", "LandReallocation"]

    # df = pd.DataFrame(X, columns=columns, index = index)

    r1svd = RankOneSvd()
    r1svd.fit_transform(X)

    # plt.figure()

    # plt.subplot(221)
    # r1svd.plot_original_matrix(matrix='A')

    # plt.subplot(222)
    # r1svd.plot_v()

    # plt.subplot(223)
    # r1svd.plot_u(transpose=True)

    # plt.subplot(224)
    # r1svd.plot_reordered_matrix(matrix='A')
    # plt.draw()

    # plt.figure()

    # plt.subplot(221)
    # r1svd.plot_original_matrix(matrix='Sr')
    # plt.subplot(222)
    # r1svd.plot_original_matrix(matrix='Sc')

    # plt.subplot(223)
    # r1svd.plot_reordered_matrix(matrix='Sr')
    # plt.subplot(224)
    # r1svd.plot_reordered_matrix(matrix='Sc')

    # plt.show()
