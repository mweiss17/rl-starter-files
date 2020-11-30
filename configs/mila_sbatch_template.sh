#! /bin/bash
# /!\ THIS FILE WILL BE PYTHON-FORMATTED: DO NOT USE CURLY-BRACKETS IN TEXT
{partition}        # Partition to use
{cpu}              # Nb. of cpus (max(unkillable)=4, max(main)=6)
{mem}              # Require memory (16GB default should be enough)
{time}             # The job will run for 4 hours
{slurm_log}        # Write the logs in /network/tmp1/<user>/covi-slurm-%j.out
{gres}             # May use GPU to get allocation
{email}            # Email Id if you want to be notified of jobs being failed/completed or started
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda deactivate
conda activate navi-generalization

export PYTHONUNBUFFERED=1

cd {code_loc}

echo $(pwd)
echo $(which python)

echo "------------------------"

# THIS FILE WILL BE APPENDED TO. DO NOT WRITE AFTER THIS LINE
