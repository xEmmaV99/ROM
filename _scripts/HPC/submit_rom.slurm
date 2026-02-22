#!/bin/bash
#SBATCH --job-name=FAM_ROM_Build
#SBATCH --output=rom_build_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=5           # Amount of jobs to run in parallel
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00       # Max time limit
# If you run into memory issues, specify it using --mem-per-cpu (e.g. --mem-per-cpu=4G)

# Load whatever
module load python/3.12
# module load fortran_compiler
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Kill everything if anything fails
set -e

MODULE_NAME=$(echo "$SCRIPT_DIR" | sed 's/.*ROM\///' | sed 's/\//./g')
python -m "$MODULE_NAME.run_pipeline" ${SLURM_NTASKS}

echo "Executing FAM runs in Parallel..."
mkdir -p "$SCRIPT_DIR/task_logs"
# Loop over the generated .sh scripts and run them in parallel using srun
# --nodes=1: Run on a single node
# --ntasks=1: Run one task per script
# --cpus-per-task=1: Allocate 1 CPU per task
# --exclusive: Ensure that each srun command gets exclusive access to that CPU
for script in src/work_dir/runfiles/*.sh; do
    # Individually log each task in a separate file
    base=$(basename "$script" .sh)
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive bash "$script" > "task_logs/${base}_${SLURM_JOB_ID}.log" 2>&1 &
done
echo "Finished!"