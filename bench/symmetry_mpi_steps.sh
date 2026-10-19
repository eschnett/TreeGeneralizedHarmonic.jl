#!/bin/bash
# The cost of a time step on 1, 2 and 4 H200s (added 2026-10-19):
# `bench/mpi_steps.jl` on the octant hole with the analytic layer, no finder
# and no fit, at two block sizes — strong scaling of the evolution alone,
# which job 578514's runs could not show because the per-chunk analysis was
# most of their wall clock (CODE.md, "Running distributed"). MPICH_jll at
# every count, and HPC-X staged at 2, to see whether its 2.5–3× there is in
# the steps. In the environments `bench/symmetry_mpi_gpu_env.sh` makes and
# job 578514 precompiled.
#
#     sbatch bench/symmetry_mpi_steps.sh

#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --job-name=treegh-mpi-steps
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treegh-mpi-steps-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0

REPO="${TREEGH_REPO:-/home/eschnetter/claude/treegh-mpi-gpu}"
ENVJ="$REPO-env-jll"
ENVH="$REPO-env-hpcx"
HPCX_MODULE="${TREEGH_HPCX_MODULE:-nvhpc-hpcx-cuda12/24.9}"
THREADS=${SLURM_CPUS_PER_TASK:-8}
STEPS=(steps_short=8 steps_long=64 backend=cuda)

echo "# $(hostname) $(date -Iseconds) $(julia --version)"
nvidia-smi -L

steps() {
    local env=$1 p=$2
    shift 2
    if [ "$p" = 0 ]; then
        julia --project="$env" -t $THREADS "$REPO/bench/mpi_steps.jl" "${STEPS[@]}" "$@"
    else
        julia --project="$env" -e '
            using MPI
            env, p, t, script, args... = ARGS
            m = MPI.mpiexec()
            x = split(get(ENV, "TREEGH_MPIEXEC_FLAGS", ""))
            run(setenv(`$m $x -n $p $(Base.julia_cmd()) --threads=$t --project=$env $script $args`,
                       m.env))' \
            "$env" "$p" "$THREADS" "$REPO/bench/mpi_steps.jl" "${STEPS[@]}" mpi=true "$@"
    fi 2>&1 | grep -E '^ranks=|ERROR|Error' || echo "FAILED: $env $p $*"
}

export HYDRA_LAUNCHER=fork
for size in "L=32 N=32" "L=48 N=48"; do
    echo "=== $size, no MPI $(date -Iseconds) ==="
    steps "$ENVJ" 0 $size
    for p in 1 2 4; do
        echo "=== $size, MPICH_jll, $p rank(s) $(date -Iseconds) ==="
        steps "$ENVJ" $p $size
    done
done
unset HYDRA_LAUNCHER
(
    module load "$HPCX_MODULE"
    # Open MPI binds a rank to one core unless told not to, and a rank's eight
    # Julia threads — rank 0's horizon find, spin and fit above all — then
    # share it (measured 2026-10-19, job 578648: 3.8× on the analysis).
    export TREEGH_MPIEXEC_FLAGS="--bind-to none"
    echo "=== L=32 N=32, HPC-X staged, 2 ranks $(date -Iseconds) ==="
    steps "$ENVH" 2 L=32 N=32
)
echo "# done $(date -Iseconds)"
