from src.ROM_basis import ROM_builder
import sys
# call this: python run_pipeline.py 10 1.0 BSkG2

if __name__ == "__main__":
    #sys.argv[0] is a path
    param=str(sys.argv[3])
    smear=float(sys.argv[2])
    num_snaps=int(sys.argv[1])

#    WAVE_FUNCTION = f'/mnt/c/users/emmav/PycharmProjects/ROM/_inputs/wf.12.12.150.150.{param}.8'
#    MF_OUT = f'/mnt/c/users/emmav/PycharmProjects/ROM/_inputs/mf.12.12.150.150.{param}.8.out'
    CODEDIR = '/gpfs/home/acad/ulb-iaa/pdemol/code/'

    WAVE_FUNCTION = f'{CODEDIR}ROM/_inputs/wf.12.12.200.200.{param}'
    MF_OUT = f'{CODEDIR}ROM/_inputs/mf.12.12.200.200.{param}.out'

    FAM_INPUT = {'w_min': 0.0,
                 'w_max': 50.0,
                 'max_num_snapshots': num_snaps,  # number of snapshots to generate
                 'smear': smear,
                 'l': 0,  # multipolarity
                 'm': 0,  # magnetic quantum number
                 }

    builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                          path_to_meanfield_out=MF_OUT,
                          FAM=FAM_INPUT,
                          tantalus_path=f'{CODEDIR}/tantalus/' #'~/code/tantalus/'
                          )

    builder.working_directory = builder.working_directory.parent.joinpath(f'work_dir.{param}.{smear}')
    builder.working_directory.mkdir(parents=True, exist_ok=True)

    builder.set_run_type("generate_runfiles")
    builder.set_build_type("equidistant_1D")
    builder.build_snapshot_basis()
