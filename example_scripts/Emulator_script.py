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
targets = np.linspace(5,40,2000)+smear*1j # example target frequencies
emul.projection_method = "PG"  # or "PG"
x,y = emul.evaluate(targets=targets)

plt.figure()
plt.plot(x.real, y)

x, y = emul.evaluate(targets=basis.omegas)
plt.scatter(x.real, y, marker='x')
plt.show()