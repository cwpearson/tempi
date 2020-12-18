#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_mpi_pingpong_1d 
#BSUB -o bench_mpi_pingpong_1d.o%J
#BSUB -e bench_mpi_pingpong_1d.e%J
#BSUB -W 00:10
#BSUB -nnodes 2

set -eou pipefail

module reset
module unload darshan-runtime
module load spectrum-mpi/10.3.1.2-20200121
module load gcc/9.3.0
module load cuda/11.0.3

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_mpi_pingpong_1d.csv

set -x

mkdir -p $SCRATCH

echo "summit pingpong-1d" > $OUT

unset TEMPI_DISABLE
export TEMPI_CONTIGUOUS_AUTO=""
echo "1nodes,2rankpernode,auto" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_AUTO
export TEMPI_CONTIGUOUS_STAGED=""
echo "1nodes,2rankpernode,staged" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_STAGED
export TEMPI_CONTIGUOUS_NONE=""
echo "1nodes,2rankpernode,fallback" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_NONE
echo "1nodes,2rankpernode,notempi" >> $OUT
export TEMPI_DISABLE=""
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_DISABLE

unset TEMPI_DISABLE
export TEMPI_CONTIGUOUS_AUTO=""
echo "2nodes,1rankpernode,auto" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_AUTO

export TEMPI_CONTIGUOUS_STAGED=""
echo "2nodes,1rankpernode,staged" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_STAGED

export TEMPI_CONTIGUOUS_NONE=""
echo "2nodes,1rankpernode,fallback" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_CONTIGUOUS_NONE

echo "2nodes,1rankpernode,notempi" >> $OUT
export TEMPI_DISABLE=""
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_DISABLE
