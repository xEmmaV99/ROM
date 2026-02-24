from src.Utils_parsers import merge_FAM_outputs
from pathlib import Path
import sys

if __name__ == "__main__":


    # folder = f'/gpfs/home/acad/ulb-iaa/pdemol/code/ROM/src/work_dir/TMP_SNAPSHOTS/'

    print("merging all files in  :" , sys.argv[1] )
    print("into folder :" , sys.argv[2] )
    
    merge_FAM_outputs(folder=sys.argv[1], output_dir=Path(sys.argv[2]))