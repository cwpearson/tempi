#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_type_commit 
#BSUB -o bench_type_commit.o%J
#BSUB -e bench_type_commit.e%J
#BSUB -W 00:15
#BSUB -nnodes 1

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc/9.3.0
module load cuda/11.0.3

DIR=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$DIR/bench_type_commit.csv

set -x

mkdir -p $DIR

echo "" > $OUT


echo "tempi" >> $OUT
unset TEMPI_DISABLE
jsrun --smpiargs="-gpu" -n 1 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-type-commit | tee -a $OUT

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
jsrun --smpiargs="-gpu" -n 1 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-type-commit | tee -a $OUT
unset TEMPI_DISABLE

