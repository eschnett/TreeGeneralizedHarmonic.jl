#!/bin/bash
# The star-damping study on Symmetry (added 2026-11-11, for TreeGRRMHD's GH-8):
# `test/star_runs.jl` runs, several per 64-core node, each a process of its own
# with `THREADS` threads, all writing under `OUT/<run>`.
#
#   sbatch [--partition=amdpreq] bench/symmetry_star_damping.sh RUNS OUT
#
# `RUNS` is a file of lines `<run name> <key=value …>`, the options of
# `test/star_runs.jl` (`#` starts a comment), where `threads=K` gives that run
# `K` threads instead of `THREADS`; `OUT` is the study's directory.
# A run that has finished (its `done`) is skipped by the script itself, and one
# with a checkpoint continues from it, so the same job requeued after a
# pre-emption or a node failure continues the study. Run from the package
# root (the rsynced copy). The submit side records each run with
# `simwatch_queued`; this script marks a run failed when its process fails.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --job-name=star-damping

set -u
RUNS=${1:?runs file}
OUT=${2:?output directory}
THREADS=${THREADS:-4}
R=$(pwd)
. "$R/bench/simwatch.sh"
export OPENBLAS_NUM_THREADS=1
echo "host $(hostname)  start $(date)  restart_count=${SLURM_RESTART_COUNT:-0}"
lscpu | grep "Model name"

# The package image, once, on this node, before the runs start together.
julia --project=. -e 'using TreeGeneralizedHarmonic' || exit 1

pids=()
names=()
while read -r name opts; do
    case "$name" in ''|'#'*) continue ;; esac
    mkdir -p "$OUT/$name"
    th=$THREADS
    [[ " $opts " =~ \ threads=([0-9]+)\  ]] && th=${BASH_REMATCH[1]}
    # shellcheck disable=SC2086
    julia --project=. --threads="$th" test/star_runs.jl $opts \
          out="$OUT/$name" name="$name" group=star-damping \
          >> "$OUT/$name/run.log" 2>&1 < /dev/null &
    pids+=($!)
    names+=("$name")
    sleep 2
done < "$RUNS"

rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    st=$?
    if [ "$st" -ne 0 ]; then
        rc=1
        simwatch_mark "$OUT/${names[$i]}" failed "exit code $st on $(hostname)"
    fi
done
echo "end $(date)  rc=$rc"
exit $rc
