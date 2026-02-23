from src.ROM_basis import ROM_builder
import sys
# call this: python run_pipeline.py 10 1.0 BSkG2

if __name__ == "__main__":
    #sys.argv[0] is a path
    param=str(sys.argv[3])
    smear=float(sys.argv[2])
    num_snaps=int(sys.argv[1])

    WAVE_FUNCTION = f'/mnt/c/users/emmav/PycharmProjects/ROM/_inputs/wf.12.12.100.100.{param}.8'
    MF_OUT = f'/mnt/c/users/emmav/PycharmProjects/ROM/_inputs/mf.12.12.100.100.{param}.8.out'

    FAM_INPUT = {'w_min': 0.0,
                 'w_max': 50.0,
                 'num_snapshots': num_snaps,  # number of snapshots to generate
                 'smear': smear,
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
