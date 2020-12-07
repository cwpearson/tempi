#!/bin/bash
#BSUB -P csc362
#BSUB -J test_measure_system 
#BSUB -o test_measure_system.o%J
#BSUB -e test_measure_system.e%J
#BSUB -W 00:10
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault

set -eou pipefail

module reset
module load gcc/7.4.0
module load cuda/11.1.1
module unload darshan-runtime

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/test_measure_system.txt

set -x

mkdir -p $SCRATCH

echo "" > $OUT

jsrun --smpiargs="-gpu" -n 1 -a 1 -g 1 -c 7 -r 1 -b rs ../../build/test/measure_system | tee -a $OUT

