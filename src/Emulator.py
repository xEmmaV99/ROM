import numpy as np
from src.Utils import _evaluate_G_numba, _evaluate_PG_numba
from src.ROM_basis import ROM_basis

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