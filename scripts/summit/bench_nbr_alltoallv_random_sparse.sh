#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_nbr_alltoallv_random_sparse 
#BSUB -o bench_nbr_alltoallv_random_sparse.o%J
#BSUB -e bench_nbr_alltoallv_random_sparse.e%J
#BSUB -W 02:00
#BSUB -nnodes 32

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc
module load cuda/11.0.3
module load nsight-systems/2020.3.1.71

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_nbr_alltoallv_random_sparse.csv

set -x

mkdir -p $SCRATCH

#echo "2nodes,6rankpernode" >> $OUT
#jsrun --smpiargs="-gpu" -n 12 -r 6 -a 1 -g 1 -c 7 -b rs \
#nsys profile -t cuda,nvtx -f true -o $SCRATCH/tempi_pattern_2n_12r_%q{OMPI_COMM_WORLD_RANK}.qdrep ../../build/bin/bench-mpi-pattern-random | tee -a $OUT

echo "" > $OUT

for nodes in 2 4 8 16 32; do
  for rpn in 6; do
    let n=$nodes*$rpn

    echo ${nodes}nodes,${rpn}rankspernode,notempi >> $OUT
    export TEMPI_PLACEMENT_NONE=""
    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-nbr-alltoallv-random-sparse | tee -a $OUT
    unset TEMPI_PLACEMENT_NONE

    echo ${nodes}nodes,${rpn}rankspernode,kahip >> $OUT
    export TEMPI_PLACEMENT_KAHIP=""
    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-nbr-alltoallv-random-sparse | tee -a $OUT
    unset TEMPI_PLACEMENT_KAHIP

  done
done

