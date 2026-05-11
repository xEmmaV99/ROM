import numpy as np
from src.Utils_emulator import evaluate_G_numba, evaluate_PG_numba, evaluate_G_SVD_numba, evaluate_PG_SVD_numba
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


    def evaluate(self, targets, svd=False):
        """
        Evaluate the emulator at the given target parameters.

        Args:
            targets: A list of frequencies at which to evaluate the emulator.
            svd: A boolean indicating whether to use Singular Value Decomposition (SVD) for the projection. Default is False.

        Returns:
            (targets, S): A tuple containing the input targets and the corresponding emulated strength.
        """

        S = np.zeros(len(targets))

        snapshot_omegas = self.basis.omegas
        F = self.basis.F
        if not svd:
            if self.projection_method == "G":
                S = evaluate_G_numba(targets, snapshot_omegas, self.basis.snapshots, F)

            elif self.projection_method == "PG":
                S, _ = evaluate_PG_numba(targets, snapshot_omegas, self.basis.snapshots, F)
        else:
            self.basis.compute_SVD() # computes SVD decomp, or nothing if already computed
            if self.projection_method=="G":
                S = evaluate_G_SVD_numba(targets, self.basis._svd_snapshot_ML, self.basis._svd_transformed_snapshots,
                                         self.basis.U, self.basis.F)
            elif self.projection_method == "PG":
                S, _ = evaluate_PG_SVD_numba(targets, self.basis.omegas, self.basis.snapshots,
                                             self.basis._svd_transformed_snapshots, self.basis.U, self.basis.F)

        return targets, S