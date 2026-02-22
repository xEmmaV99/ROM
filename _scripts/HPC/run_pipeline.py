from src.ROM_basis import ROM_builder
import sys

if __name__ == "__main__":
    WAVE_FUNCTION = f'/mnt/c/users/emmav/PycharmProjects/ROM/_inputs/wf.12.12.100.100.BSkG2.8'
    MF_OUT = f'/mnt/c/users/emmav/PycharmProjects/ROM/_inputs/mf.12.12.100.100.BSkG2.8.out'

    FAM_INPUT = {'w_min': 0.0,
                 'w_max': 40.0,
                 'num_snapshots': int(sys.argv[1]),  # number of snapshots to generate
                 'smear': 1.0,
                 'l': 0,  # multipolarity
                 'm': 0,  # magnetic quantum number
                 }

    builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                          path_to_meanfield_out=MF_OUT,
                          FAM=FAM_INPUT,
                          tantalus_path='~/code/tantalus/'
                          )

    builder.set_run_type("generate_runfiles")
    builder.set_build_type("equidistant_1D")
    builder.build_snapshot_basis()