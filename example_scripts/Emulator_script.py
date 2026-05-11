"""
Example script for emulator application
"""

from src.ROM_basis import ROM_basis
from src.Emulator import Emulator
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
CWD = Path(os.getcwd())

# Load emulator basis
path_to_snapshot = CWD.parent.joinpath('_outputs/xy.12.12.30.30.0.0.BSkG2.16.xy')
basis = ROM_basis()
basis.load(path_to_snapshot=path_to_snapshot)

# Create emulator
emul = Emulator(basis=basis)

smear = 1.0
targets = np.linspace(-40,40,2000)+smear*1j # example target frequencies
emul.projection_method = "G"  # or "PG"

x,y = emul.evaluate(targets=targets)

plt.figure()
plt.plot(x.real, y)

# now also apply the svd decomp:
x, y = emul.evaluate(targets=targets, svd=True)

plt.plot(x.real, y, label='SVD')

# now also add symmetries
emul.basis.expand_by_symmetry(which="all")
# need to recompute the SVD because the snapshot matrix has changed
emul.basis.compute_SVD(force_calc=True, cutoff=0.001)

x,y = emul.evaluate(targets=targets, svd=True)
plt.plot(x.real, y, label='symmetry expanded')


plt.legend()

plt.show()