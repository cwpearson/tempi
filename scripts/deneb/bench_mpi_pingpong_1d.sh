#! /bin/bash

set -eou pipefail

MPIRUN="$HOME/software/openmpi-4.0.5/bin/mpirun -n 2"
export MV2_USE_CUDA=1
# MPIRUN="$HOME/software/mvapich2-2.3.4/bin/mpirun -n 2"

DIR=$HOME/sync_work/tempi_results/deneb
OUT=$DIR/bench_mpi_pack.csv

set -x

mkdir -p $DIR

echo "" > $OUT

echo "auto" >> $OUT
export TEMPI_CONTIGUOUS_AUTO=""
$MPIRUN ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_AUTO

echo "staged" >> $OUT
export TEMPI_CONTIGUOUS_STAGED=""
$MPIRUN ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_STAGED

echo "fallback" >> $OUT
unset TEMPI_DISABLE
$MPIRUN ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
$MPIRUN ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_DISABLE


