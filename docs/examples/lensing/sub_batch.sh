#!/usr/bin/env bash
CONFIG_NAMES=(  
    "t30_z10"
    "t30_z10_so"
            )

# Constants
RUNNAME='raytracing'
RUNFILE='/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/fin_test/run.py'

# resource request
PARTITION="MIKO"
CONDA_ENV="cosmo1"
GRES="gpu:1"
OUTDIR="/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/fin_test"

# Create a tmp SLURM script
# Determine the directory of the currently running script
mkdir -p "${OUTDIR}/tmp"

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    TMP_SLURM_SCRIPT=$(mktemp "${OUTDIR}/tmp/slurm_script.XXXXXXXXX")
    echo "TMP_SLURM_SCRIPT: $TMP_SLURM_SCRIPT"
    # log file
    TIMESTAMP=$(date +%d%H%M%S)
    LOGPATH="${OUTDIR}/tmp/${RUNNAME}_${CONFIG_NAME}_${TIMESTAMP}.log"
    echo "LOGPATH: $LOGPATH"

    # Write script to the temporary SLURM script
    cat > $TMP_SLURM_SCRIPT << EOL
#!/bin/bash

#SBATCH -p $PARTITION
#SBATCH -t 72:00:00
#SBATCH --job-name ${RUNNAME}_${CONFIG_NAME}
#SBATCH --output $LOGPATH
#SBATCH --gres=$GRES
#SBATCH -A phy230056p
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junzhez@andrew.cmu.edu

echo "Running on \$(hostname)"
echo "Running configuration: $CONFIG_NAME"

source ~/.bashrc
echo "Sourced the bashrc file"
echo ""

module load cuda; module use /opt/packages/spack/share/spack/lmod/linux-rhel8-x86_64/gcc/11.2.0; module load anaconda3/2023.09-0-z4cmm6p
echo "Setup script executed successfully"
echo "nvcc"
nvcc --version
echo "conda"
conda config --show envs_dirs
echo ""

conda activate $CONDA_ENV
echo "Conda environment activated: $CONDA_ENV"
echo ""

python $RUNFILE $CONFIG_NAME

# Remove the temporary SLURM script
rm $TMP_SLURM_SCRIPT
EOL

    # Submit the SLURM job using sbatch
    sbatch $TMP_SLURM_SCRIPT
done
