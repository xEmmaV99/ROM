import numpy as np
from src.Utils_emulator import _evaluate_G_numba, _evaluate_PG_numba
from src.ROM_basis import ROM_basis

class Emulator:
    def __init__(self, basis: ROM_basis):
        self.basis = basis # ROM_basis object
        self.projection_method = "PG" # or "G"

    def evaluate(self, targets):
        import time
        start = time.time()

        S = np.zeros(len(targets))

        X = self.basis.snapshots[:, 0, :]
        Y = self.basis.snapshots[:, 1, :]
        snapshot_omegas = self.basis.omegas
        F = self.basis.F

        if self.projection_method == "G":
            S = _evaluate_G_numba(targets, snapshot_omegas, X, Y, F)

        elif self.projection_method == "PG":
            S = _evaluate_PG_numba(targets, snapshot_omegas, X, Y, F)


        print("Elapsed emulation time: ", time.time() - start, " s")
        return targets, S