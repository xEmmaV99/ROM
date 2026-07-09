from numba import njit, prange
import numpy as np


@njit(fastmath=True, cache=True)
def cost_numba(omega, alpha, FdF, MXdF, XMMX, F_norm, snapshot_omegas):
    """
    Computes the cost function for the given parameters.

    Args:
        omega: frequency at which to evaluate the cost function
        alpha: solution vector of the linear system ( A alpha = b ) evaluated at the frequency omega
        FdF: precomputed scalar value representing the inner product of F and F^dagger
        MXdF: precomputed vector representing (M X)^dagger F
        XMMX: precomputed vector representing (MX)^dagger (MX)
        F_norm: scaling factor for the cost function, typically the norm of F
        snapshot_omegas: vector of frequencies corresponding to the snapshots used in the ROM basis

    Returns:
        The computed cost function value for omega.
    """
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

@njit(inline='always',parallel=True, fastmath=True, cache=True)
def evaluate_G_numba(targets, snapshot_omegas, snapshot_matrix, F):
    """
    Evaluation of the Galerkin projected FAM equations.

    Args:
        targets: The target frequencies at which to evaluate the strength.
        snapshot_omegas: snapshot frequencies corresponding to the ROM basis, (num_snapshots,).
        snapshot_matrix: snapshot matrix mathcal(X) = [X, Y] of the ROM basis (num_snapshots, 2, num_entries).
        F: F vector (num_entries,).

    Returns:
        S: The evaluated strength at the target frequencies (num_targets,).

    """
    X, Y = snapshot_matrix[:, 0, :], snapshot_matrix[:, 1, :]

    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    S = np.zeros(n_targets, dtype=np.float64)

    overlap_weights = np.sum(np.conj(F) * np.concatenate((X, Y), axis=1), axis=1)
    D = overlap_weights.conj()
    M = X.conj() @ X.T - Y.conj() @ Y.T

    C_vals = np.zeros((n_targets, n_snaps), dtype=np.complex128)
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
            C_vals[w] = np.linalg.solve(A, -D)
            dot_val = 0.0 + 0.0j
            for k in range(n_snaps):
                dot_val += C_vals[w,k] * overlap_weights[k]
            S[w] = -2 * dot_val.imag / np.pi
    return S, C_vals

@njit(inline='always',parallel=True, fastmath=True, cache=True) #DEBUG EMMA
def evaluate_PG_numba(targets, snapshot_omegas, snapshot_matrix, F):
    """
    Evaluation of the Galerkin projected FAM equations.

    Args:
        targets: The target frequencies at which to evaluate the strength.
        snapshot_omegas: snapshot frequencies corresponding to the ROM basis, (num_snapshots,).
        snapshot_matrix: snapshot matrix mathcal(X) = [X, Y] of the ROM basis (num_snapshots, 2, num_entries).
        F: F vector (num_entries,).

    Returns:
        S: The evaluated strength at the target frequencies (num_targets,).
        C_vals: The coefficients C for each target frequency (num_targets, num_snapshots), used for calculating the error estimator (cost function) for greedy sampling
    """
    X, Y = snapshot_matrix[:, 0, :], snapshot_matrix[:, 1, :]

    MXdF = np.concatenate((X.conj(), -Y.conj()), axis=1) @ F
    FdF = np.vdot(F, F)
    FdMX = np.conj(MXdF)
    XMMX = np.conj(X) @ X.T + np.conj(Y) @ Y.T

    overlap_weights = np.sum(np.conj(F)*np.concatenate((X, Y), axis=1),axis=1)

    n_targets = len(targets)
    n_snaps = len(snapshot_omegas)
    S = np.zeros(n_targets, dtype=np.float64)

    C_vals = np.zeros((n_targets, n_snaps), dtype=np.complex128)
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
        C_vals[w, :] = C

        dot_val = 0.0 + 0.0j
        for k in range(n_snaps):
            dot_val += C[k] * overlap_weights[k]

        S[w] = -2 * dot_val.imag / np.pi

    return S, C_vals


@njit(parallel=True, fastmath=True, cache=True)
def evaluate_G_SVD_numba(targets, snapshot_ML, transformed_snapshots, U, F):
    """
    Evaluation of the Galerkin projected FAM equations including the SVD.

    Args:
        targets: The target frequencies at which to evaluate the strength.
        snapshot_ML: precalculated transformed matrix (see expression in the thesis for details)
        transformed_snapshots: transformed snapshot matrix of the ROM basis
        U: U matrix from the SVD of the snapshot matrix (num_snaps, num_modes).
        F: F vector (num_entries,).

    Returns:
        S: The evaluated strength at the target frequencies (num_targets,).
    """
    num_targets = len(targets)

    X = np.ascontiguousarray(transformed_snapshots[:, 0, :])
    Y = np.ascontiguousarray(transformed_snapshots[:, 1, :])
    X_ML = np.ascontiguousarray(snapshot_ML[:, 0, :])
    Y_ML = np.ascontiguousarray(snapshot_ML[:, 1, :])

    F_vec = F.ravel()
    D = np.sum(F_vec * np.concatenate((np.conj(X), np.conj(Y)), axis=1), axis=1)
    overlap_weights = np.conj(D)

    M = (np.conj(X) @ X.T) - (np.conj(Y) @ Y.T)
    ML = (np.conj(X) @ X_ML.T) - (np.conj(Y) @ Y_ML.T)

    num_modes = U.shape[1]
    num_snaps = U.shape[0]
    U_sum = np.zeros(num_modes, dtype=np.complex128)
    C_SVD = np.zeros((num_targets, U.shhape[1]), dtype=np.complex128)
    for i in range(num_modes):
        for l in range(num_snaps):
            U_sum[i] += U[l, i]

    S = np.zeros(num_targets, dtype=np.float64)
    for w in prange(num_targets):
        omega_target = targets[w]
        A = (ML - omega_target * M) - np.outer(D, U_sum)
        b = -D
        C = np.linalg.solve(A, b)
        C_SVD[w] = C
        dot_val = np.dot(C, overlap_weights)
        S[w] = -2 * dot_val.imag / np.pi

    return S, C_SVD


@njit(parallel=True, fastmath=True, cache=True)
def evaluate_PG_SVD_numba(targets, snapshot_omegas_orig, snapshot_matrices_orig, snapshot_matrices, U, F):
    """
    Evaluation of the Petrov-Galerkin projected FAM equations including the SVD.

    Args:
        targets: The target frequencies at which to evaluate the strength.
        snapshot_omegas_orig:  snapshot frequencies corresponding to the ROM basis, (num_snapshots,).
        snapshot_matrices_orig: snapshot matrix mathcal(X) = [X, Y] of the ROM basis (num_snapshots, 2, num_entries).
        snapshot_matrices: transformed/truncated snapshot matrix
        U: U matrix from the SVD of the snapshot matrix (num_snaps, num_modes).
        F: F vector (num_entries,).

    Returns:
        S: The evaluated strength at the target frequencies (num_targets,).
        C_SVD: The coefficients C for each target frequency (num_targets, num_modes), used for calculating the error estimator (cost function) for greedy sampling
    """
    Ns = U.shape[0]
    Nr = U.shape[1]
    n_targets = len(targets)

    F= F.ravel()
    X = np.ascontiguousarray(snapshot_matrices_orig[:, 0, :])
    Y = np.ascontiguousarray(snapshot_matrices_orig[:, 1, :])

    XMMX = np.dot(np.conj(X), X.T) + np.dot(np.conj(Y), Y.T)
    FdF = np.vdot(F, F).real
    MXdF = np.concatenate((np.conj(X), -np.conj(Y)), axis=1) @ F
    MXdF_conj = np.conj(MXdF)

    Xu = np.ascontiguousarray(snapshot_matrices[:, 0, :])
    Yu = np.ascontiguousarray(snapshot_matrices[:, 1, :])
    overlap_weights = np.sum(np.conj(F)*np.concatenate((Xu, Yu), axis=1),axis=1)

    SumU = np.sum(U, axis=0)
    SumU_conj = np.conj(SumU)

    Ud = np.ascontiguousarray(np.conj(U).T)
    O = snapshot_omegas_orig
    O_conj = np.conj(O)

    inner_v0 = O * MXdF_conj
    v2_0 = np.dot(Ud, inner_v0)
    v2_1 = np.dot(Ud, MXdF_conj)

    b0 = FdF * SumU_conj - v2_0
    b1 = v2_1

    temp_P0 = np.empty((Ns, Ns), dtype=np.complex128)
    for i in range(Ns):
        for j in range(Ns):
            temp_P0[i, j] = XMMX[i, j] * O_conj[i] * O[j]
    P0 = np.dot(Ud, np.dot(temp_P0, U))

    temp_P1 = np.empty((Ns, Ns), dtype=np.complex128)
    for i in range(Ns):
        for j in range(Ns):
            temp_P1[i, j] = XMMX[i, j] * O_conj[i]
    P1 = np.dot(Ud, np.dot(temp_P1, U))

    P3 = np.dot(Ud, np.dot(XMMX, U))

    A0 = np.zeros((Nr, Nr), dtype=np.complex128)
    A_w = np.zeros((Nr, Nr), dtype=np.complex128)

    for i in range(Nr):
        si_c = SumU_conj[i]
        v0_i = v2_0[i]
        v1_i = v2_1[i]
        for j in range(Nr):
            A0[i, j] = P0[i, j] + FdF * si_c * SumU[j] - v0_i * SumU[j] - si_c * np.conj(v2_0[j])
            A_w[i, j] = -P1[i, j] + v1_i * SumU[j]

    A0 = 0.5 * (A0 + np.conj(A0.T))
    P3 = 0.5 * (P3 + np.conj(P3.T))

    A_w_conj_T = np.ascontiguousarray(np.conj(A_w.T))

    S = np.zeros(n_targets, dtype=np.float64)
    C_SVD = np.zeros((n_targets, Nr), dtype=np.complex128)

    for w in prange(n_targets):
        wt = targets[w]
        wt_c = np.conj(wt)
        wt_abs_sq = (wt * wt_c).real

        b_PG = b0 + wt * b1
        A_PG = A0 + (wt * A_w) + (wt_c * A_w_conj_T) + (wt_abs_sq * P3)

        c = np.linalg.solve(A_PG, b_PG)
        C_SVD[w, :] = c

        dot_val = np.dot(c, overlap_weights)
        S[w] = -2.0 * dot_val.imag / np.pi

    return S, C_SVD