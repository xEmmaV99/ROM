from src.Create_ROM_basis import ROM_builder, ROM_basis
from src.Utils import combine_FAM_output
from pathlib import Path

WAVE_FUNCTION = '/mnt/c/users/emmav/PycharmProjects/ROM/_input/wf.12.12.30.30'
MF_OUT = '/mnt/c/users/emmav/PycharmProjects/ROM/_input/mf.12.12.30.30.out'


rom_builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                          path_to_meanfield_out=MF_OUT)

rom_builder.build_snapshot_basis_static(max_workers=8)

path_to_snapshot = rom_builder.path_to_snapshot
# path_to_snapshot = 'C:/users/emmav/PycharmProjects/ROM/_output/xy.12.12.30.30.0.8.xy'


basis = ROM_basis()
basis.load(path_to_snapshot=path_to_snapshot)

print(basis.omegas)

# FAM_output = Path(r'C:\Users\emmav\PycharmProjects\ROM\_output')
# combine_FAM_output(directory='C:\\Users\emmav\PycharmProjects\ROM\src\work_dir\TMP_SNAPSHOTS', output_dir=FAM_output)
