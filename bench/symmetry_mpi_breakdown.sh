#!/bin/bash
# Where a time step goes on 1, 2 and 4 H200s (added 2026-10-19):
# `bench/mpi_breakdown.jl` on the octant mesh of `bench/mpi_steps.jl` (7.6 M
# points) — no MPI, MPICH_jll at 2 and 4 ranks, HPC-X at 2 and 4 staged and
# direct, and HPC-X with Open MPI's default core binding switched off — and
# then one `test/octant_runs.jl` tracked `:fitted` run under HPC-X without the
# binding, whose chunk time job 578514 measured at 2.5–3× MPICH's. In the
# environments `bench/symmetry_mpi_gpu_env.sh` makes.
#
#     sbatch bench/symmetry_mpi_breakdown.sh

#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --job-name=treegh-mpi-breakdown
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treegh-mpi-breakdown-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0

REPO="${TREEGH_REPO:-/home/eschnetter/claude/treegh-mpi-gpu}"
ENVJ="$REPO-env-jll"
ENVH="$REPO-env-hpcx"
HPCX_MODULE="${TREEGH_HPCX_MODULE:-nvhpc-hpcx-cuda12/24.9}"
OUT="/mnt/beegfs/eschnetter/claude/treegh-mpi-breakdown/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
THREADS=${SLURM_CPUS_PER_TASK:-8}
SIZE=(L=32 N=32 roots=4 radii=16,8,4 backend=cuda reps=5)

echo "# $(hostname) $(date -Iseconds) $(julia --version)"
nvidia-smi -L
which nsys || true

# `$1` the environment, `$2` the ranks (0: no MPI), `$3` extra launcher
# arguments, then the script and its arguments.
launch() {
    local env=$1 p=$2 extra=$3
    shift 3
    if [ "$p" = 0 ]; then
        julia --project="$env" -t $THREADS "$@"
    else
        julia --project="$env" -e '
            using MPI
            env, p, t, extra, script, args... = ARGS
            m = MPI.mpiexec()
            x = split(extra)
            run(setenv(`$m $x -n $p $(Base.julia_cmd()) --threads=$t --project=$env $script $args`,
                       m.env))' \
            "$env" "$p" "$THREADS" "$extra" "$@" mpi=true
    fi
}
breakdown() {
    local name=$1 env=$2 p=$3 extra=$4
    shift 4
    echo "=== $name $(date -Iseconds) ==="
    launch "$env" "$p" "$extra" "$REPO/bench/mpi_breakdown.jl" "${SIZE[@]}" "$@" 2>&1 |
        grep -vE 'Warning|^ *[│└┌]|^ *$' | tail -40 || echo "FAILED: $name"
}

breakdown "no MPI, 1 GPU" "$ENVJ" 0 ""
export HYDRA_LAUNCHER=fork
for p in 2 4; do
    breakdown "MPICH_jll, $p ranks" "$ENVJ" $p ""
done
unset HYDRA_LAUNCHER
(
    module load "$HPCX_MODULE"
    for p in 2 4; do
        breakdown "HPC-X staged, $p ranks" "$ENVH" $p ""
        breakdown "HPC-X direct, $p ranks" "$ENVH" $p "" deviceaware=true
    done
    breakdown "HPC-X staged, 2 ranks, --bind-to none" "$ENVH" 2 "--bind-to none"

    echo "=== octant :fitted run, HPC-X, 2 ranks, --bind-to none $(date -Iseconds) ==="
    launch "$ENVH" 2 "--bind-to none" "$REPO/test/octant_runs.jl" case=ks interior=fitted \
        L=32 N=32 roots=4 radii=16,8,4 t_end=3/2 chunk=1/2 cfl=1/2 backend=cuda \
        out="$OUT/octant-hpcx-n2-unbound" > "$OUT/octant-hpcx-n2-unbound.log" 2>&1 ||
        echo "octant run FAILED"
    grep -E '^t =|^done' "$OUT/octant-hpcx-n2-unbound.log" |
        sed -E 's/^(t = +[0-9.]+ +wall +[0-9.]+ s).*/\1/' || true
)
echo "# done $(date -Iseconds)"
