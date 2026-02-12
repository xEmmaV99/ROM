import numpy as np
import numba as nb
from src.ROM_basis import ROM_basis


@nb.njit(parallel=True, fastmath=True, cache=True)
def _evaluate_G_numba(targets, snapshot_omegas, D, M, overlap_weights):
    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    S = np.zeros(n_targets, dtype=np.float64)
    b = -D.copy()
    for w in nb.prange(n_targets):
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

@nb.njit(parallel=True, fastmath=True, cache=True)
def _evaluate_PG_numba(targets, snapshot_omegas, FdF, FdMX, MXdF, XMMX, overlap_weights):
    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    S = np.zeros(n_targets, dtype=np.float64)

    for w in nb.prange(n_targets):
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

class Emulator:
    def __init__(self, basis: ROM_basis):
        self.basis = basis # ROM_basis object
        self.projection_method = "G" # or "G"

    def evaluate(self, targets):
        import time
        start = time.time()

        S = np.zeros(len(targets))

        X = self.basis.snapshots[:, 0, :]
        Y = self.basis.snapshots[:, 1, :]
        snapshot_omegas = self.basis.omegas
        F = self.basis.F

        # We need <F|X_i> + <F|Y_i>.
        # Since F, X, Y are complex, use sum(conj(F)*X, axis=1)
        overlap_X = np.sum(np.conj(F) * X, axis=1)
        overlap_Y = np.sum(np.conj(F) * Y, axis=1)
        overlap_weights = overlap_X + overlap_Y

        if self.projection_method == "G":
            D = (X + Y).conj() @ F
            M = X.conj() @ X.T - Y.conj() @ Y.T

            # Flatten D if it came out as (N, 1)
            if D.ndim > 1: D = D.ravel()

            S = _evaluate_G_numba(targets, snapshot_omegas, D, M, overlap_weights)

        elif self.projection_method == "PG":
            # Precompute PG matrices
            diff_XY = X - Y
            FdF = F.conj().T @ F  # Scalar (or 1x1 matrix)
            MXdF = diff_XY.conj() @ F  # Shape (N_snaps,)
            FdMX = MXdF.conj()  # Shape (N_snaps,)
            XMMX = X.conj() @ X.T + Y.conj() @ Y.T  # Shape (N_snaps, N_snaps)
            if np.ndim(FdF) != 0:
                FdF = FdF.item()

            S = _evaluate_PG_numba(targets, snapshot_omegas, FdF, FdMX, MXdF, XMMX, overlap_weights)


        print("Elapsed emulation time: ", time.time() - start, " s")
        return targets, S