"""
Example script for basis creation
"""
from src.ROM_basis import ROM_builder
import os
from pathlib import Path
CWD = Path(os.getcwd())

WAVE_FUNCTION = str(CWD.parent.joinpath('_inputs/wf.12.12.30.30'))
MF_OUT = str(CWD.parent.joinpath('_inputs/mf.12.12.30.30.out'))
TANTALUS = '/home/emma/code/tantalus/'
FAM_INPUT = {'w_min': 5.0,
             'w_max': 40.0,
             'max_num_snapshots' : 10,
             'smear': 1.0,
             'l': 0, # multipolarity
             'm': 0,  # magnetic quantum number
             }

rom_builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                          path_to_meanfield_out=MF_OUT,
                          FAM=FAM_INPUT,
                          tantalus_path=TANTALUS)

rom_builder.build_type = 'equidistant' # 'greedy'
rom_builder.data.max_num_snapshots = 35 #override

# LOAD if it already exists
if rom_builder.build_type == 'greedy':
    path_to_snapshot = CWD.parent.joinpath('_outputs/xy.12.12.30.30.0.0.BSkG2.16.xy')
    try:
        rom_builder.basis.load(path_to_snapshot)
        print("Loaded existing ROM basis from ", path_to_snapshot)
    except Exception as e:
        print("No existing ROM basis found at ", path_to_snapshot, ". Building new basis.")

    rom_builder.build_snapshot_basis()
elif rom_builder.build_type == 'equidistant':
    rom_builder.build_snapshot_basis(max_workers=8)
else:
    rom_builder.build_snapshot_basis()

print("ROM basis saved at ", rom_builder.path_to_snapshot)