#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_mpi_send 
#BSUB -o bench_mpi_send.o%J
#BSUB -e bench_mpi_send.e%J
#BSUB -W 02:00
#BSUB -nnodes 2

#"BSUB -alloc_flags gpudefault"

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc
module load cuda/11.0.3

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_mpi_send.csv

set -x

mkdir -p $SCRATCH

# CSV header
echo "2nodes,1rankpernode" > $OUT
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 42 -b rs ../../build/bin/bench-mpi-send | tee -a $OUT

echo "1nodes,2rankpernode" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 21 -b rs ../../build/bin/bench-mpi-send | tee -a $OUT

