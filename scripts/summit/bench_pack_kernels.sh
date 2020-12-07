#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_pack_kernels 
#BSUB -o bench_pack_kernels.o%J
#BSUB -e bench_pack_kernels.e%J
#BSUB -W 00:10
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module reset
module load gcc/7.4.0
module load cuda/11.1.1

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_pack_kernels.csv

set -x

mkdir -p $SCRATCH

echo "" > $OUT

# had to add disable_gpu_hooks, was getting a CUDA library hook error at cudaFree
# december 4 2020
# module reset
# cuda 11.1.1
# gcc 7.4.0
jsrun --smpiargs="-disable_gpu_hooks" -n 1 -a 1 -g 1 -c 7 -r 1 -b rs ../../build/bin/bench-pack-kernels | tee -a $OUT

