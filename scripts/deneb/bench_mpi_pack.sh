#! /bin/bash

set -eou pipefail

export MV2_USE_CUDA=1

# MPIRUN="$HOME/software/openmpi-4.0.5/bin/mpirun -n 1"
MPIRUN="$HOME/software/mvapich2-2.3.4/bin/mpirun -n 1"

DIR=$HOME/sync_work/tempi_results/deneb
OUT=$DIR/bench_mpi_pack.csv

set -x

mkdir -p $DIR

echo "" > $OUT


echo "tempi" >> $OUT
unset TEMPI_DISABLE
$MPIRUN ../../build/bin/bench-mpi-pack | tee -a $OUT

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
$MPIRUN ../../build/bin/bench-mpi-pack | tee -a $OUT
unset TEMPI_DISABLE


