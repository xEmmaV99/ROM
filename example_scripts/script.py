from src.ROM_basis import ROM_builder, ROM_basis
from src.Emulator import Emulator
import numpy as np
import matplotlib.pyplot as plt

WAVE_FUNCTION = '/mnt/c/users/emmav/PycharmProjects/ROM/_input/wf.12.12.30.30'
MF_OUT = '/mnt/c/users/emmav/PycharmProjects/ROM/_input/mf.12.12.30.30.out'
FAM_INPUT = {'w_min': 5.0,
             'w_max': 40.0,
             'num_snapshots' : 12,
             'smear': 1.0,
             'l': 2 , # multipolarity
             'm': 0,  # magnetic quantum number
             }

rom_builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                          path_to_meanfield_out=MF_OUT,
                          FAM=FAM_INPUT)

rom_builder.build_type = 'greedy'
rom_builder.data.num_snapshots = 14 #override

# LOAD if it already exists
path_to_snapshot = 'C:/users/emmav/PycharmProjects/ROM/_output/xy.12.12.30.30.2.0.xy'
rom_builder.basis.load(path_to_snapshot)

#rom_builder.build_snapshot_basis_static(max_workers=8)
rom_builder.build_snapshot_basis()

path_to_snapshot = rom_builder.path_to_snapshot


basis = ROM_basis()
basis.load(path_to_snapshot=path_to_snapshot)
## OR
basis = rom_builder.basis

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