import numpy as np

from src.ROM_basis import ROM_basis
from parsers import FAMtalus

path_to_data = "_inputs/Mg24_DDPC1/"

# first, load in the basis
freqs = FAMtalus.identify_frequencies(path_to_data)
freqs = freqs[[int(k) for k in np.linspace(0, len(freqs)-1, 5)]]
freqs = np.concatenate((freqs, [23.8, 32.4, 22.5, 20.2, 17.4, 15.5, 27.2, 12.9, 35.7, 28.9]))
omegas, matrices, F = FAMtalus.parse_from_folder(path_to_data, freqs)

print(omegas)

""" code """
basis = ROM_basis()
basis.load(omegas=omegas, snapshots=matrices, F=F)

basis.next_snapshot(d_omega=0.1)

targets = np.linspace(0, 40, 2000)+1.0j
targets, S = basis.evaluate(targets)

import matplotlib.pyplot as plt
plt.plot(targets, S)
plt.show()


