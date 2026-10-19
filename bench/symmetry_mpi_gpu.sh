#!/bin/bash
# The multi-GPU run of TreeGeneralizedHarmonic's MPI port on Symmetry
# (added 2026-10-19): one tracked `:fitted` Kerr-Schild hole on the octant,
# `test/octant_runs.jl`, on one H200 serially and then over MPI with one rank
# per GPU — MPICH_jll (host staging) and HPC-X's CUDA-aware Open MPI
# through host staging and with the
# device buffers handed to MPI directly (`deviceaware=true`). Every run's
# `octant.csv` and `records.csv` are compared with the serial run's
# (`bench/octant_compare.jl`: the same text but for the last digits of L2
# sums), and the wall clock at each chunk is in each run's log. TreeAMR's
# `bench/symmetry_mpi_gpu.sh` is the pattern; see "Running distributed" in
# CODE.md.
#
#     sbatch bench/symmetry_mpi_gpu.sh
#
# It runs MPICH_jll at 1 rank and HPC-X and MPICH_jll at every rank count
# from 2 up to the GPUs allocated, in steps of doubling — 4 GPUs on `h200q`
# by default, since `h200debugq`'s QOS allows the group only 2 GPUs.
#
# The two environments (`$REPO-env-jll`, `$REPO-env-hpcx`) are made and
# instantiated on the login node beforehand, with `JULIA_PKG_PRECOMPILE_AUTO=0`;
# they are precompiled here, on the GPU node.

#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --job-name=treegh-mpi-gpu
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treegh-mpi-gpu-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# MPICH segfaults in its first collective without this (symmetry-hpc).
export MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0

REPO="${TREEGH_REPO:-/home/eschnetter/claude/treegh-mpi-gpu}"
ENVJ="$REPO-env-jll"
ENVH="$REPO-env-hpcx"
HPCX_MODULE="${TREEGH_HPCX_MODULE:-nvhpc-hpcx-cuda12/24.9}"
OUT="/mnt/beegfs/eschnetter/claude/treegh-mpi-gpu/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
THREADS=${SLURM_CPUS_PER_TASK:-8}
NGPU=$(nvidia-smi -L | wc -l)
RANKS=()
for ((p = 2; p <= NGPU; p *= 2)); do RANKS+=($p); done
# Four chunks: the first pays the compilation, the other three are the cost.
RUN=(case=ks interior=fitted L=32 N=32 roots=4 radii=16,8,4 t_end=2 chunk=1/2
     cfl=1/2 backend=cuda)

echo "# $(hostname) $(date -Iseconds) $(julia --version)"
lscpu | grep 'Model name' || true
nvidia-smi -L
nvidia-smi topo -m || true

echo "=== precompiling (MPICH_jll) $(date -Iseconds) ==="
julia --project="$ENVJ" -e 'using Pkg; Pkg.precompile()'
echo "=== precompiling (HPC-X) $(date -Iseconds) ==="
( module load "$HPCX_MODULE" && julia --project="$ENVH" -e 'using Pkg; Pkg.precompile()' )

# One GPU, no MPI: the reference.
echo "=== serial, 1 GPU $(date -Iseconds) ==="
julia --project="$ENVJ" -t $THREADS "$REPO/test/octant_runs.jl" "${RUN[@]}" \
    out="$OUT/serial" > "$OUT/serial.log" 2>&1 || echo "serial FAILED"
tail -n 3 "$OUT/serial.log"

# `-n P` ranks under the launcher of the environment's MPI binary, with its
# environment (MPI.jl's `mpiexec()`), one GPU and THREADS threads a rank.
mpirun_octant() {
    local env=$1 p=$2 name=$3
    shift 3
    echo "=== $name: $p rank(s) $(date -Iseconds) ==="
    julia --project="$env" -e '
        using MPI
        env, p, t, script, args... = ARGS
        m = MPI.mpiexec()
        x = split(get(ENV, "TREEGH_MPIEXEC_FLAGS", ""))
        cmd = `$m $x -n $p $(Base.julia_cmd()) --threads=$t --project=$env $script $args`
        run(setenv(cmd, m.env))' \
        "$env" "$p" "$THREADS" "$REPO/test/octant_runs.jl" "${RUN[@]}" mpi=true "$@" \
        out="$OUT/$name" > "$OUT/$name.log" 2>&1 || echo "$name FAILED"
    tail -n 3 "$OUT/$name.log"
}

# MPICH_jll: hydra would launch its proxies through `srun`; on one node it
# can fork them.
export HYDRA_LAUNCHER=fork
for p in 1 "${RANKS[@]}"; do
    mpirun_octant "$ENVJ" $p "jll-n$p"
done
unset HYDRA_LAUNCHER

# HPC-X, CUDA-aware: staged and direct.
(
    module load "$HPCX_MODULE"
    # Open MPI binds a rank to one core unless told not to, and a rank's eight
    # Julia threads — rank 0's horizon find, spin and fit above all — then
    # share it (measured 2026-10-19, job 578648: 3.8× on the analysis).
    export TREEGH_MPIEXEC_FLAGS="--bind-to none"
    for p in "${RANKS[@]}"; do
        mpirun_octant "$ENVH" $p "hpcx-n$p-staged" deviceaware=false
        mpirun_octant "$ENVH" $p "hpcx-n$p-direct" deviceaware=true
    done
)

echo "=== comparison with the serial run $(date -Iseconds) ==="
julia "$REPO/bench/octant_compare.jl" "$OUT/serial" \
    $(for d in "$OUT"/jll-n* "$OUT"/hpcx-n*; do [ -d "$d" ] && echo "$d"; done) || true

echo "=== wall clock at each chunk ==="
for f in "$OUT"/*.log; do
    echo "--- $(basename "$f" .log)"
    grep -E '^t = |^done' "$f" | sed -E 's/^(t = +[0-9.]+ +wall +[0-9.]+ s).*/\1/' || true
done
echo "# done $(date -Iseconds)"
