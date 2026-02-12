from src.Create_ROM_basis import ROM_builder, ROM_basis
from src.Utils import combine_FAM_output
from pathlib import Path

WAVE_FUNCTION = '/mnt/c/users/emmav/PycharmProjects/ROM/_input/wf.12.12.30.30'
MF_OUT = '/mnt/c/users/emmav/PycharmProjects/ROM/_input/mf.12.12.30.30.out'
FAM_INPUT = {'w_min': 0.0,
             'w_max': 40.0,
             'num_snapshots' : 4,
             'smear': 0.25,
             'l': 0 , # multipolarity
             'm': 0,  # magnetic quantum number
             }

path_to_snapshot = 'C:/users/emmav/PycharmProjects/ROM/_output/xy.12.12.30.30.0.8.xy'

if Path(path_to_snapshot).exists():
    print("Snapshot file already exists, skipping ROM basis construction.")
else:
    rom_builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                              path_to_meanfield_out=MF_OUT,
                              FAM=FAM_INPUT)

    rom_builder.build_snapshot_basis_static(max_workers=8)

    path_to_snapshot = rom_builder.path_to_snapshot


basis = ROM_basis()
basis.load(path_to_snapshot=path_to_snapshot)

print(basis.omegas)

# FAM_output = Path(r'C:\Users\emmav\PycharmProjects\ROM\_output')
# combine_FAM_output(directory='C:\\Users\emmav\PycharmProjects\ROM\src\work_dir\TMP_SNAPSHOTS', output_dir=FAM_output)
