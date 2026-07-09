import numpy as np
import matplotlib.pyplot as plt

from src.ROM_basis import ROM_basis
from parsers import FAMtalus

path_to_data = "inputs/Mg24_DDPC1/"

# first, load in the basis
freqs = FAMtalus.identify_frequencies(path_to_data)
# keep 5 evenly spaced frequencies
indices = np.linspace(0, len(freqs) - 1, 5, dtype=int)
freqs = freqs[indices]
# add specified frequencies
extra_freqs = np.array([23.8, 32.4, 22.5, 20.2, 17.4])

freqs = np.concatenate((freqs, extra_freqs))
omegas, matrices, F = FAMtalus.parse_from_folder(path_to_data, freqs)

print(omegas)

""" code """
basis = ROM_basis()
basis.load(omegas=omegas, snapshots=matrices, F=F)

basis.next_snapshot(d_omega=0.1)

targets = np.linspace(0, 40, 2000)+1.0j
targets, S = basis.evaluate(targets)




