#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_pack 
#BSUB -o bench_pack.o%J
#BSUB -e bench_pack.e%J
#BSUB -W 00:10
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc/7.4.0
module load cuda/11.1.1
module load nsight-systems/2020.4.1.144

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_pack.csv

set -x

mkdir -p $SCRATCH

echo "" > $OUT

unset TEMPI_DISABLE
jsrun --smpiargs="-gpu" -n 1 -a 1 -g 1 -c 7 -r 1 -b rs ../../build/bin/bench-pack | tee -a $OUT
#jsrun --smpiargs="-gpu" -n 1 -a 1 -g 1 -c 7 -r 1 -b rs nsys profile -t cuda,nvtx -o $SCRACH/bench_pack -f true ../../build/bin/bench-pack | tee -a $OUT

