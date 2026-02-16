from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_G_numba(targets, snapshot_omegas, D, M, overlap_weights):
    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    S = np.zeros(n_targets, dtype=np.float64)
    b = -D.copy()
    for w in prange(n_targets):
        omega_target = targets[w]
        delta = snapshot_omegas - omega_target

        # if omega is part of the snapshot... G breaks...
        min_dist_idx = -1
        min_dist = 1e20
        for i in range(n_snaps):
            dist = np.abs(delta[i])
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = i
        if min_dist < 1e-12:
            val = overlap_weights[min_dist_idx]
            S[w] = -2 * val.imag / np.pi
        else:
            A = np.zeros((n_snaps, n_snaps), dtype=np.complex128)
            for i in range(n_snaps):
                for j in range(n_snaps):
                    A[i, j] = -D[i] + M[i, j] * delta[j]

            C = np.linalg.solve(A, b)
            dot_val = 0.0 + 0.0j
            for k in range(n_snaps):
                dot_val += C[k] * overlap_weights[k]
            S[w] = -2 * dot_val.imag / np.pi
    return S

@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_PG_numba(targets, snapshot_omegas, FdF, FdMX, MXdF, XMMX, overlap_weights):
    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    S = np.zeros(n_targets, dtype=np.float64)

    for w in prange(n_targets):
        omega_target = targets[w]
        delta = snapshot_omegas - omega_target
        b = np.zeros(n_snaps, dtype=np.complex128)
        for i in range(n_snaps):
            b[i] = FdF - (delta[i] * FdMX[i])
        A = np.zeros((n_snaps, n_snaps), dtype=np.complex128)
        for i in range(n_snaps):
            d_conj_i = np.conj(delta[i])
            for j in range(n_snaps):
                val = FdF
                val -= delta[i] * FdMX[i]
                val += d_conj_i * delta[j] * XMMX[i, j]
                val -= np.conj(delta[j]) * MXdF[j]
                A[i, j] = val

        C = np.linalg.solve(A, b)

        dot_val = 0.0 + 0.0j
        for k in range(n_snaps):
            dot_val += C[k] * overlap_weights[k]

        S[w] = -2 * dot_val.imag / np.pi

    return S
