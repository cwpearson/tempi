#!/bin/bash
#BSUB -P csc362
#BSUB -J measure_system 
#BSUB -o measure_system.o%J
#BSUB -e measure_system.e%J
#BSUB -W 00:20
#BSUB -nnodes 2
#BSUB -alloc_flags gpudefault

set -eou pipefail

module reset
module load gcc/9.3.0
module load cuda/11.1.1
module unload darshan-runtime

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/measure_system.txt

set -x

mkdir -p $SCRATCH

echo "" > $OUT


export TEMPI_CACHE_DIR=$SCRATCH
#1 node, 2 ranks/node
jsrun --smpiargs="-gpu" -n 2 -a 1 -g 1 -c 7 -r 2 -b rs ../../build/bin/measure-system | tee -a $OUT

#2 node, 1 ranks/node
jsrun --smpiargs="-gpu" -n 2 -a 1 -g 1 -c 7 -r 1 -b rs ../../build/bin/measure-system | tee -a $OUT

