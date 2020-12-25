#! /bin/bash

set -eou pipefail

export MV2_USE_CUDA=1

MPIRUN="$HOME/local/openmpi/bin/mpirun -n 2"

DIR=$HOME/sync_work/tempi_results/deneb
OUT=$DIR/bench_send.csv

set -x

mkdir -p $DIR

echo "" > $OUT


echo "oneshot" >> $OUT
unset TEMPI_DISABLE
export TEMPI_DATATYPE_ONESHOT=""
$MPIRUN ../../build/bin/bench-mpi-ireduce | tee -a $OUT
unset TEMPI_DATATYPE_ONESHOT

echo "device" >> $OUT
unset TEMPI_DISABLE
export TEMPI_DATATYPE_DEVICE=""
$MPIRUN ../../build/bin/bench-mpi-ireduce | tee -a $OUT
unset TEMPI_DATATYPE_DEVICE

echo "auto" >> $OUT
unset TEMPI_DISABLE
export TEMPI_DATATYPE_AUTO=""
$MPIRUN ../../build/bin/bench-mpi-ireduce | tee -a $OUT
unset TEMPI_DATATYPE_AUTO

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
$MPIRUN ../../build/bin/bench-mpi-ireduce | tee -a $OUT
unset TEMPI_DISABLE


