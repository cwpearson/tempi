#!/bin/bash

set -eou pipefail

DIR=$HOME/sync/tempi_results
OUT=$DIR/bench_halo_exchange.csv

set -x

mkdir -p $DIR

echo "" > $OUT

#MPIRUN="$HOME/software/mvapich2-2.3.4/bin/mpirun"
MPIRUN="$HOME/software/openmpi-4.0.5/bin/mpirun"

nodes=1;
  for rpn in 2; do
    let n=$nodes*$rpn

    if   [ $n ==   1 ]; then X=512
    elif [ $n ==   2 ]; then X=645
    elif [ $n ==   4 ]; then X=813
    elif [ $n ==   6 ]; then X=930
    fi

    echo "${nodes}nodes,${rpn}rankspernode,tempi" >> $OUT
    export TEMPI_PLACEMENT_KAHIP=""
    $MPIRUN -n $n ../../build/bin/bench-halo-exchange 100 $X | tee -a $OUT
    unset TEMPI_PLACEMENT_KAHIP

#    echo "${nodes}nodes,${rpn}rankspernode,notempi" >> $OUT
#    export TEMPI_DISABLE=""
#    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-halo-exchange 5 $X | tee -a $OUT
#    unset TEMPI_DISABLE
  done
done

