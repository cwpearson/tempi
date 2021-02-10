#!/bin/bash
#BSUB -P csc362
#BSUB -J bench_astaroth 
#BSUB -o bench_astaroth.o%J
#BSUB -e bench_astaroth.e%J
#BSUB -W 02:00
#BSUB -nnodes 63

set -eou pipefail

module reset
module unload darshan-runtime
module load gcc/9.3.0
module load cuda/11.0.3
module load nsight-systems/2020.3.1.71

SCRATCH=/gpfs/alpine/scratch/cpearson/csc362/tempi_results
OUT=$SCRATCH/bench_astaroth.csv
export TEMPI_CACHE_DIR=$SCRATCH

set -x

mkdir -p $SCRATCH

echo "" > $OUT

for nodes in 1 2 4 8 16 32 64 128 256 512; do
  for rpn in 1 2 6; do
    let n=$nodes*$rpn

    if   [ $n ==    1 ]; then X=256  Y=256  Z=256
    elif [ $n ==    2 ]; then X=512  Y=256  Z=256
    elif [ $n ==    4 ]; then X=1024 Y=256  Z=256
    elif [ $n ==    6 ]; then X=768  Y=512  Z=256
    elif [ $n ==    8 ]; then X=1024 Y=512  Z=256
    elif [ $n ==   12 ]; then X=1536 Y=512  Z=256
    elif [ $n ==   16 ]; then X=1024 Y=512  Z=512
    elif [ $n ==   24 ]; then X=1536 Y=1024 Z=256
    elif [ $n ==   32 ]; then X=1536 Y=1024 Z=256
    elif [ $n ==   48 ]; then X=1536 Y=1024 Z=512
    elif [ $n ==   64 ]; then X=2048 Y=1024 Z=512
    elif [ $n ==   96 ]; then X=3072 Y=1024 Z=512
    elif [ $n ==  128 ]; then X=2048 Y=1024 Z=1024
    elif [ $n ==  192 ]; then X=3072 Y=2048 Z=512
    elif [ $n ==  256 ]; then X=4096 Y=1024 Z=1024
    elif [ $n ==  384 ]; then X=3072 Y=2048 Z=1024
    elif [ $n ==  512 ]; then X=4096 Y=2048 Z=1024
    elif [ $n ==  768 ]; then X=6144 Y=2048 Z=1024
    elif [ $n == 1536 ]; then X=6144 Y=4096 Z=1024
    elif [ $n == 3072 ]; then X=6144 Y=4096 Z=2048
    fi

    echo "${nodes}nodes,${rpn}rankspernode,tempi" >> $OUT
    export TEMPI_PLACEMENT_KAHIP=""
    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-halo-exchange 100 $X $Y $Z | tee -a $OUT
    #jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs nsys profile -t cuda,nvtx -f true -o $SCRATCH/bench_halo_exchange_${nodes}n_${rpn}rpn_${X}x_%q{OMPI_COMM_WORLD_RANK} ../../build/bin/bench-halo-exchange 10 $X
    unset TEMPI_PLACEMENT_KAHIP

#    echo "${nodes}nodes,${rpn}rankspernode,notempi" >> $OUT
#    export TEMPI_DISABLE=""
#    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-halo-exchange 5 $X | tee -a $OUT
#    unset TEMPI_DISABLE
  done
done

