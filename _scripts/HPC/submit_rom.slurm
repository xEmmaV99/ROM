#!/bin/bash
#SBATCH --account=nuclr
#SBATCH --partition=debug
#SBATCH --job-name=FAM_ROM_Build
#SBATCH --output=rom_build_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4           # Amount of jobs to run in parallel
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00       # Max time limit
# If you run into memory issues, specify it using --mem-per-cpu (e.g. --mem-per-cpu=4G)

SMEAR=1.0 # 0.1 0.25 0.5 0.75 1.0 1.5
PARAM="BSkG2" # "BSkG2" "BSkG4" "BSkG5"

# restore my module list 
module restore tantalus-hdf5-serial

ROM_DIR="/gpfs/home/acad/ulb-iaa/pdemol/code/ROM/"
SCRIPT_DIR="${ROM_DIR}/_scripts/HPC/"
MERGE_DIR="${ROM_DIR}/_scripts/"

# Kill everything if anything fails
set -e

# # MODULE_NAME=$(echo "$SCRIPT_DIR" | sed 's/.*ROM\///' | sed 's/\//./g')
# python "$SCRIPT_DIR/run_pipeline.py" ${SLURM_NTASKS} ${SMEAR} ${PARAM}

# echo "Executing FAM runs in Parallel..."
# echo "$SCRIPT_DIR/task_logs"
# mkdir -p "$SCRIPT_DIR/task_logs"
# # Loop over the generated .sh scripts and run them in parallel using srun
# # --nodes=1: Run on a single node
# # --ntasks=1: Run one task per script
# # --cpus-per-task=1: Allocate 1 CPU per task
# # --exclusive: Ensure that each srun command gets exclusive access to that CPU
# for script in src/work_dir.${PARAM}.${SMEAR}/runfiles/*.sh; do
#     # Individually log each task in a separate file
#     base=$(basename "$script" .sh)
#     srun --nodes=1 --ntasks=1 --cpus-per-task=1  --exclusive bash "$script" > 'test.stdout' &
# done

# wait 

TMP_SNAP=$ROM_DIR/src/work_dir.$PARAM.$SMEAR/TMP_SNAPSHOTS/
OUTDIR=$ROM_DIR/_outputs/$PARAM.$SMEAR/

mkdir -p $OUTDIR

python $MERGE_DIR/merge.py $TMP_SNAP $OUTDIR

echo "Finished!"
