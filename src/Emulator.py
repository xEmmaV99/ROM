import numpy as np
from src.Utils_emulator import _evaluate_G_numba, _evaluate_PG_numba
from src.ROM_basis import ROM_basis

class Emulator:
    """
    Loads in the Emulator and handles evaluation.

    Attributes:
        basis (ROM_basis): The underlying ROM basis object used for calculations.
        projection_method (str): The method used for projection.
            Can be "PG" (Petrov-Galerkin) or "G" (Galerkin).
    """
    def __init__(self, basis: ROM_basis):
        """
        Initialize the Emulator.

        Args:
            basis: An instance of ROM_basis.
        """
        self.basis = basis # ROM_basis object
        self.projection_method = "PG" # or "G"

    def evaluate(self, targets):
        """
        Evaluate the emulator at the given target parameters.

        Args:
            targets: A list of frequencies at which to evaluate the emulator.

        Returns:
            (targets, S): A tuple containing the input targets and the corresponding emulated strength.
        """

        S = np.zeros(len(targets))

        X = self.basis.snapshots[:, 0, :]
        Y = self.basis.snapshots[:, 1, :]
        snapshot_omegas = self.basis.omegas
        F = self.basis.F

        if self.projection_method == "G":
            S = _evaluate_G_numba(targets, snapshot_omegas, X, Y, F)

        elif self.projection_method == "PG":
            S, _ = _evaluate_PG_numba(targets, snapshot_omegas, X, Y, F)

        return targets, S