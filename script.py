from tests.utils.helpers import Paths

from src.CreateROM import CreateROM
import os
from pathlib import Path
DATA_FOLDER = "$HOME/projects/tantalus_private/ssh_dump/MEANFIELD_OUT/wf.12.12.30.30"


rom_builder = CreateROM(path_to_meanfield_wf=DATA_FOLDER,
                        p=12,
                        n=12,
                        w_min=0.0,
                        w_max=30.0,
                        smear=0.5,
                        snapshots=2)

rom_builder.build_snapshot_basis_static()