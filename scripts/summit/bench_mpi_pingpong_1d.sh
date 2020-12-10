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
module load gcc/7.4.0
module load cuda/11.0.3

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_mpi_pingpong_1d.csv

set -x

mkdir -p $SCRATCH

echo "summit pingpong-1d" > $OUT

echo "1nodes,2rankpernode,tempi" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
echo "1nodes,2rankpernode,notempi" >> $OUT
export TEMPI_DISABLE=""
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_DISABLE

echo "2nodes,1rankpernode,tempi" >> $OUT
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
echo "2nodes,1rankpernode,notempi" >> $OUT
export TEMPI_DISABLE=""
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT
unset TEMPI_DISABLE

#echo "2nodes,2rankpernode" >> $OUT
#jsrun --smpiargs="-gpu" -n 4 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT

#echo "2nodes,3rankpernode" >> $OUT
#jsrun --smpiargs="-gpu" -n 6 -r 3 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT

#echo "2nodes,4rankpernode" >> $OUT
#jsrun --smpiargs="-gpu" -n 8 -r 4 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT

#echo "2nodes,5rankpernode" >> $OUT
#jsrun --smpiargs="-gpu" -n 10 -r 5 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT

#echo "2nodes,6rankpernode" >> $OUT
#jsrun --smpiargs="-gpu" -n 12 -r 6 -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-mpi-pingpong-1d | tee -a $OUT

