import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def cost_numba(omega, alpha, FdF, MXdF, XMMX, F_norm, snapshot_omegas):
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

    return 1/F_norm*(term_FF.real + term_XX.real + real_cross)