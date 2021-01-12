#! /bin/bash

set -eou pipefail

export MV2_USE_CUDA=1
export MV2_USE_GDRCOPY=0
export MV2_USE_GPUDIRECT_RDMA=0
export MV2_USE_GDR=0
export MV2_GPUDIRECT_GDRCOPY_LIB=$HOME/software/gdrcopy-2.1/lib64/libgdrapi.so

DIR=$HOME/sync_work/tempi_results/deneb
OUT=$DIR/bench_mpi_pack.csv

set -x

mkdir -p $DIR

echo "" > $OUT


echo "tempi" >> $OUT
unset TEMPI_DISABLE
# ../../build/mpiexec -n 1 nsys profile -t cuda,nvtx ../../build/bin/bench-mpi-pack | tee -a $OUT
../../build/mpiexec -n 1 ../../build/bin/bench-mpi-pack | tee -a $OUT

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
../../build/mpiexec -n 1 ../../build/bin/bench-mpi-pack | tee -a $OUT
unset TEMPI_DISABLE


