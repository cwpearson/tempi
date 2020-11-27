#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_mpi_pack 
#BSUB -o bench_mpi_pack.o%J
#BSUB -e bench_mpi_pack.e%J
#BSUB -W 01:00
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc/7.4.0
module load cuda/11.1.1

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_mpi_pack.csv

set -x

mkdir -p $SCRATCH

echo "" > $OUT

echo "tempi" >> $OUT
unset TEMPI_DISABLE
jsrun --smpiargs="-gpu" -n 1 -a 1 -g 1 -c 7 -r 1 -b rs ../../build/bin/bench-mpi-pack | tee -a $OUT

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
jsrun --smpiargs="-gpu" -n 1 -a 1 -g 1 -c 7 -r 1 -b rs ../../build/bin/bench-mpi-pack | tee -a $OUT
unset TEMPI_DISABLE

