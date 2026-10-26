#!/bin/bash
# The two environments `bench/symmetry_mpi_gpu.sh` runs in (added
# 2026-10-19), made on Symmetry's login node: the package by path, CUDA, MPI
# and what `test/octant_runs.jl` loads, with the package's own `[sources]`
# pins — one with MPICH_jll and one with HPC-X's CUDA-aware Open MPI as the
# system binary. Instantiated without precompiling: the login node is Intel
# and the job precompiles on the GPU node.
#
#     bash bench/symmetry_mpi_gpu_env.sh

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_PKG_PRECOMPILE_AUTO=0
REPO="${TREEGH_REPO:-/home/eschnetter/claude/treegh-mpi-gpu}"
HPCX_MODULE="${TREEGH_HPCX_MODULE:-nvhpc-hpcx-cuda12/24.9}"

make_env() {
    local dir=$1 binary=$2
    mkdir -p "$dir"
    cat > "$dir/Project.toml" <<EOF
[deps]
ApparentHorizonFinder = "2326d305-03d0-4260-ba27-fc7d1edfc55d"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
IMEXRungeKutta = "6f46d5f7-e15e-41aa-a098-eed9d1a63ad8"
KorzynskiSpin = "dddeb715-7fc8-4947-8935-98683755b697"
SpacetimeMetrics = "77e61b41-605e-43a6-b70f-04b7dcd1f572"
TreeIOHDF5 = "49af10b2-b28a-4989-bafa-527ac30d3246"
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
MPIPreferences = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
TreeAMR = "91bc416d-811c-44c1-9bc4-b9c8bbdbbd08"
TreeGeneralizedHarmonic = "ec251ae6-5b3e-4df5-a9c7-7e3f01cdd68c"

[sources]
TreeGeneralizedHarmonic = {path = "$REPO"}
ApparentHorizonFinder = {rev = "main", url = "https://github.com/eschnett/ApparentHorizonFinder.jl"}
IMEXRungeKutta = {rev = "main", url = "https://github.com/eschnett/IMEXRungeKutta.jl"}
KorzynskiSpin = {rev = "main", url = "https://github.com/eschnett/KorzynskiSpin.jl"}
SpacetimeMetrics = {rev = "main", url = "https://github.com/eschnett/SpacetimeMetrics.jl"}
EOF
    julia --project="$dir" -e "
        using Pkg
        Pkg.Registry.update()
        Pkg.instantiate()
        using MPIPreferences
        $binary
        Pkg.instantiate()"
}

make_env "$REPO-env-jll" 'MPIPreferences.use_jll_binary()'
(
    module load "$HPCX_MODULE"
    ENVH="$REPO-env-hpcx"
    make_env "$ENVH" 'MPIPreferences.use_system_binary()'
    # HDF5_jll follows MPIPreferences' ABI and loads OpenMPI_jll, whose own
    # libopen-pal does not load beside HPC-X's libraries ("undefined symbol:
    # pmix_framework_names", job 578514): point OpenMPI_jll's products at
    # HPC-X's, so that the process holds one MPI library.
    OMPI="$(dirname "$(dirname "$(which mpiexec)")")"
    sed -i 's/^\[deps\]/[deps]\nOpenMPI_jll = "fe0851c0-eecd-5654-98d4-656369965a5c"/' \
        "$ENVH/Project.toml"
    {
        echo
        echo "[OpenMPI_jll]"
        for lib in libmpi libmpi_mpifh libmpi_usempi_ignore_tkr libmpi_usempif08; do
            echo "${lib}_path = \"$OMPI/lib/$lib.so\""
        done
        echo "mpiexec_path = \"$OMPI/bin/mpiexec\""
    } >> "$ENVH/LocalPreferences.toml"
    julia --project="$ENVH" -e 'using Pkg; Pkg.resolve()'
)
echo "environments ready"
