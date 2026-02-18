import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def cost_numba(omega, alpha, FdF, MXdF, XMMX, snapshot_omegas):
    alpha_flat = alpha.ravel()
    MXdF_flat = MXdF.ravel()
    n = len(alpha_flat)

    v = np.zeros(n, dtype=np.complex128)
    sum_alpha = 0.0 + 0.0j

    for i in range(n):
        val = alpha_flat[i]
        sum_alpha += val
        v[i] = val * (snapshot_omegas[i] - omega)

    c_F = 1.0 - sum_alpha

    term_FF = (np.abs(c_F) ** 2) * FdF

    term_XX = 0.0 + 0.0j
    for i in range(n):
        mat_vec_dot = 0.0 + 0.0j
        for j in range(n):
            mat_vec_dot += XMMX[i, j] * v[j]

        term_XX += np.conj(v[i]) * mat_vec_dot

    cross_part = 0.0 + 0.0j
    for i in range(n):
        cross_part += np.conj(v[i]) * MXdF_flat[i]

    term_cross_A = c_F * cross_part
    real_cross = 2.0 * term_cross_A.real

    return term_FF.real + term_XX.real + real_cross

@njit(parallel=True, fastmath=True, cache=True)
def _evaluate_PG_numba_coef(targets, snapshot_omegas, FdF, FdMX, MXdF, XMMX):
    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    C_list = np.zeros((len(targets), n_snaps), dtype=np.complex128)

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
        C_list[w, :] = C.flatten()

    return C_list