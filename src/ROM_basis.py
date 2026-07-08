import numpy as np
try:
    from src.Utils import cost_numba, evaluate_PG_numba, evaluate_G_numba, evaluate_G_SVD_numba, evaluate_PG_SVD_numba
    from numba import prange
except ImportError:
    print("Numba not installed.")


class ROM_basis:
    """
    Class to represent the ROM_basis

    Attributes:
        omegas: The frequencies (omegas) at which the snapshots were taken, a 1D array (num_snapshots,) of complex values.
        snapshots: The snapshot data, typically a 3D array (num_snapshots, 2, entries), first dimension refers to the snapshot frequency, second dimension is for X and Y and the third dimension contains the matrix elements.
        F: The F vector associated with the snapshots (stacked F20, F02)
        U: SVD transformation matrix
    """
    def __init__(self):
        """
        Initialise the ROM_basis with empty attributes. The actual data will be loaded using the load method.
        """
        self.omegas = None
        self.snapshots = None
        self.F = None
        self.U = None


        # ROM SETTINGS
        self.settings = {"convergence_criterion": 0.0, #threshold of the greedy snapshot cost cutoff
                         "threshold": 0.1, #selecting snapshots below 0.1 relative cost
                         "smear": 1, # target msearing
                         "max_smearing": 20.0, # maximum smearing to consider for new snapshots in 2D greedy
                         "w_min": 0, "w_max": 40,
                         }
        self.build_type = 'greedy'
        self.projection_method = "PG"

    def is_loaded(self):
        return self.omegas is not None and self.snapshots is not None


    def load(self, omegas, snapshots, F):
        """
        Load the ROM_basis with the provided omegas and snapshots, the user is responsible for ensuring that the omegas and snapshots are consistent and valid.
        Args:
            omegas: Nx1 array of complex frequencies at which the snapshots were taken
            snapshots: Nx2xM array of snapshot data, where N is the number of snapshots, 2 corresponds to X and Y, and M is the number of matrix elements
        """
        self.omegas = omegas
        self.snapshots = snapshots
        self.F = F


    def compute_SVD(self, cutoff=0.001, force_calc=False):
        """
            Computes the Singular Value Decomposition (SVD) of the snapshot data

        Args:
            cutoff: the cutoff threshold for singular values (relative to the largest one)
            force_calc: if True, always re-calculate the SVD
        """
        if self.U is None or force_calc:
            nx, ny, nz = self.snapshots.shape
            X_flat = self.snapshots.reshape(nx, ny * nz).T
            U, S, Vh = np.linalg.svd(X_flat, full_matrices=False)  # full_matrices false when not square
            U_snap = Vh.T @ np.diag(1 / S)
            self.U = U_snap[:,  S / S[0] > cutoff]

            ### precalc for projection
            # PG
            self._svd_transformed_snapshots = np.einsum('ji,jkl->ikl', self.U, self.snapshots)
            # G
            self._svd_snapshot_ML = np.einsum('l,lab,lk->kab', self.omegas, self.snapshots, self.U)


    def expand_by_symmetry(self, which="all"):
        """
        Adding snapshots related by symmetry
        X(-w) = Y(w) and Y(-w) = X(w) symmetry
        X(w*) = X*(w) and Y(w*) = Y*(w) symmetry
        X(-w*) = Y*(w) and Y(-w*) = X*(w) symmetry

        Args:
            which: string specifying which symmetires to add (default "" adds all, "pm" adds +/- omega symmetry, "cc" adds complex conjugation symmetry)

        """
        if which == "all" or which == "pm":
            print(self.omegas.shape)
            omega = np.concatenate((self.omegas, -self.omegas))

            tmp = np.zeros_like(self.snapshots)
            tmp[:, 0, :] = self.snapshots[:, 1, :]  # X(-w) = Y(w)
            tmp[:, 1, :] = self.snapshots[:, 0, :]  # Y(-w) = X(w)
            snapshots = np.stack((self.snapshots, tmp)).reshape((-1, self.snapshots.shape[1], self.snapshots.shape[2]))

            self.omegas = omega # update here so we also include it in the next step if required
            self.snapshots = snapshots

        #X(w*) = X(w).conj() and Y(w*) = Y(w).conj()  symmetry
        if which == "all" or which == "cc":
            print("Adding free snapshots for cc omega symmetry")
            omega = np.concatenate(
                (self.omegas, self.omegas.conj()))  # This also includes the last one since i am conjugating omega-new
            tmp = np.zeros_like(self.snapshots)
            tmp[:, 0, :] = self.snapshots[:, 0, :].conj()
            tmp[:, 1, :] = self.snapshots[:, 1, :].conj()
            snapshots = np.stack((self.snapshots, tmp)).reshape((-1, self.snapshots.shape[1], self.snapshots.shape[2]))

        self.snapshots = snapshots
        self.omegas = omega


    def next_snapshot(self, d_omega=None):
        """
        Builds the snapshot basis by launching FAM runs according to the specified build type and run type.
        The method handles the entire process of generating FAM runfiles, executing them, and processing the output to construct the ROM basis.

        Returns:
            Prints new snapshot location and corresponding cost
        """
        if not self.is_loaded():
            raise ValueError("ROM_basis is not loaded. Please load the basis before building the snapshot basis.")

        else:
            snapshots = self.snapshots
            snapshot_omegas = self.omegas
            F = self.F


        if self.build_type == 'greedy':
            # iterative basis creation
            if d_omega is None:
                W_scan = np.linspace(self.settings['w_min'], self.settings['w_max'], num=2000) + self.settings['smear'] * 1.j
            else:

                W_scan = np.arange(self.settings['w_min'], self.settings['w_max'], d_omega) + self.settings['smear'] * 1.j
            FdF = F.conj().T @ F
            COSTS = np.zeros(len(W_scan))

            X = snapshots[:, 0, :]
            Y = snapshots[:, 1, :]
            MXdF = np.concatenate((X.conj(), -Y.conj()), axis=1) @ F
            XMMX = X.conj() @ X.T + Y.conj() @ Y.T

            _, alphas = evaluate_PG_numba(W_scan, snapshot_omegas, snapshots, F)

            for idx in prange(len(W_scan)):
                omega_test = W_scan[idx]
                alpha = alphas[idx]

                COSTS[idx] = cost_numba(omega=omega_test,
                                  alpha=alpha,
                                  FdF=FdF, MXdF=MXdF, XMMX=XMMX, F_norm=np.linalg.norm(F),
                                  snapshot_omegas=snapshot_omegas)

            max_cost_idx = np.argmax(COSTS)
            print('New snapshot: ', W_scan[max_cost_idx], '\nCost: ', COSTS[max_cost_idx])
            return W_scan[max_cost_idx], COSTS[max_cost_idx]


        elif self.build_type == 'greedy_2D':
            # Define the W_scan @ the target smearing
            W_scan = np.linspace(d.w_min, d.w_max, num=2000) + d.smear * 1.j

            FdF = F.conj().T @ F
            COSTS = np.zeros(len(W_scan))

            X = snapshots[:, 0, :]
            Y = snapshots[:, 1, :]

            MXdF = np.concatenate((X.conj(), -Y.conj()), axis=1) @ F
            XMMX = X.conj() @ X.T + Y.conj() @ Y.T

            _, alphas = evaluate_PG_numba(W_scan, snapshot_omegas, snapshots, F)

            for idx in prange(len(W_scan)):
                omega_test = W_scan[idx]
                alpha = alphas[idx]

                COSTS[idx] = cost_numba(omega=omega_test,
                                        alpha=alpha,
                                        FdF=FdF, MXdF=MXdF, XMMX=XMMX, F_norm=np.linalg.norm(F),
                                        snapshot_omegas=snapshot_omegas)

            max_cost_idx = np.argmax(COSTS)

            # now, scan this omega value along the complex axis, and find the "sub"-optimal omega value
            H_scan = np.real(W_scan[max_cost_idx]) + np.linspace(1, self.greedy_2D_settings["max_smearing"],200)*1j*d.smear
            COSTS_H = np.zeros(len(H_scan))
            _, alphas = evaluate_PG_numba(H_scan, snapshot_omegas, snapshots, F)
            for idx in prange(len(H_scan)):
                omega_test = H_scan[idx]
                alpha = alphas[idx]
                COSTS_H[idx] = cost_numba(omega=omega_test,
                                        alpha=alpha,
                                        FdF=FdF, MXdF=MXdF, XMMX=XMMX, F_norm=np.linalg.norm(F),
                                        snapshot_omegas=snapshot_omegas)

            # find the maximal cost in H but below a threshold
            COSTS_H = COSTS_H / np.max(COSTS_H)  # normalise wrt "highest" one
            COSTS_H = np.where(COSTS_H > self.greedy_2D_settings["threshold"], 0.0, COSTS_H)
            if np.all(COSTS_H == 0.0):
                print("No new snapshot found above threshold, adding the one with largest smearing")
                max_cost_idx_H = -1
            else:
                max_cost_idx_H = np.argmax(COSTS_H)
            SUB_OPTIMAL_OMEGA = H_scan[max_cost_idx_H]
            print("Selected : ", SUB_OPTIMAL_OMEGA)

        else:
            raise ValueError("Error")


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

        if not self.is_loaded():
            raise ValueError("ROM_basis is not loaded. Please load the basis before building the snapshot basis.")
        else:
            snapshots = self.snapshots
            snapshot_omegas = self.omegas
            F = self.F

        if not svd:
            if self.projection_method == "G":
                S, _ = evaluate_G_numba(targets, snapshot_omegas, snapshots, F)
            elif self.projection_method == "PG":
                S, _ = evaluate_PG_numba(targets, snapshot_omegas, snapshots, F)
        else:
            self.compute_SVD()  # computes SVD decomp, or nothing if already computed
            if self.projection_method == "G":
                S, _ = evaluate_G_SVD_numba(targets, self._svd_snapshot_ML, self._svd_transformed_snapshots, self.U, F)
            elif self.projection_method == "PG":
                S, _ = evaluate_PG_SVD_numba(targets, snapshot_omegas, snapshots, self._svd_transformed_snapshots, self.U, self.F)

        return targets, S