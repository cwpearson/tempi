#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_mpi_pack 
#BSUB -o bench_mpi_pack.o%J
#BSUB -e bench_mpi_pack.e%J
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc
module load cuda/11.0.3

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_mpi_pack.csv

set -x

mkdir -p $SCRATCH

# CSV header
echo "x,y,z,nodes,ranks-per-node,self-per-node,colo-per-node,remote-per-node,self,colo,remote,first (s),total (s)" > $OUT

jsrun --smpiargs="-gpu" -n 1 -a 1 -g 1 -c 42 -r 1 -b rs ../../build/bin/bench-mpi-pack | tee -a $OUT


