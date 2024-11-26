#!/usr/bin/env bash

# ## strict bash mode
set -eEuo pipefail

# ## run via
# ##    . load_env.sh
# ## otherwise a subshell is used and the PATH / variable adjustments are only made there.
# module reset
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load lsfm-init-miniconda/1.0.0

# ## get name of environment as specified in environment.yml
environment_file=environment.pytorch_lipophilicity.yml
conda_env_name=$(sed -ne 's/^name: \(.*\)$/\1/p' ${environment_file:?})
echo "################## Load (and set up) conda env ${conda_env_name:?}"

# ## install env if it does not exist
if ! conda info --envs | grep -Eq "^${conda_env_name:?} "; then
    echo "- Create conda environment ${conda_env_name:?}"

    # if [[ -z "$MPICC" ]]; then
    #     MPICC=$(which mpicc)
    # fi
    # echo "MPICC: $MPICC"
    conda env create -f ${environment_file:?} || { echo "Environment creation failed!"; exit 1; }
fi

# ## activate env
# shellcheck disable=SC2063
if [[ $(conda info --envs | grep -c '*') != 0 ]]; then
    echo "- Deactivate current environment and activate ${conda_env_name:?}"
    conda deactivate
    conda activate "${conda_env_name:?}"
else
    echo "- Activate ${conda_env_name:?}"
    conda activate "${conda_env_name:?}"
fi
