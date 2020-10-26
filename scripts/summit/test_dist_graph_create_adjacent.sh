#!/bin/bash
#BSUB -P csc362
#BSUB -J test_dist_graph_create_adjacent 
#BSUB -o test_dist_graph_create_adjacent.o%J
#BSUB -e test_dist_graph_create_adjacent.e%J
#BSUB -W 00:05
#BSUB -nnodes 2

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc
module load cuda/11.1.0
module load nsight-systems/2020.3.1.71

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_mpi_pattern.csv

set -x

mkdir -p $SCRATCH

echo "" > $OUT



echo "1nodes,2rankpernode"
jsrun --smpiargs="-gpu" -n 2 -r 2 -a 1 -g 1 -c 7 -b rs ../../build/test/dist_graph_create_adjacent

echo "2nodes,1rankpernode"
jsrun --smpiargs="-gpu" -n 2 -r 1 -a 1 -g 1 -c 7 -b rs ../../build/test/dist_graph_create_adjacent

echo "2nodes,6rankpernode"
jsrun --smpiargs="-gpu" -n 12 -r 6 -a 1 -g 1 -c 7 -b rs ../../build/test/dist_graph_create_adjacent

