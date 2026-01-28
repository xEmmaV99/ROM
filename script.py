from src.CreateROM import CreateROM
from src.Utils import combine_FAM_output
from pathlib import Path

WAVE_FUNCTION = "$HOME/projects/tantalus_private/ssh_dump/MEANFIELD_OUT/wf.12.12.30.30"

rom_builder = CreateROM(path_to_meanfield_wf=WAVE_FUNCTION,
                        p=12,
                        n=12,
                        w_min=0.0,
                        w_max=30.0,
                        smear=100,
                        snapshots=2)

rom_builder.build_snapshot_basis_static()

# FAM_output = Path(r'C:\Users\emmav\PycharmProjects\ROM\_output')
# combine_FAM_output(directory='C:\\Users\emmav\PycharmProjects\ROM\src\work_dir\TMP_SNAPSHOTS', output_dir=FAM_output)
