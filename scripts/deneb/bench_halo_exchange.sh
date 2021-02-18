#!/bin/bash

set -eou pipefail

DIR=$HOME/sync_work/tempi_results
OUT=$DIR/bench_halo_exchange.csv

set -x

mkdir -p $DIR

echo "" > $OUT

#MPIRUN="$HOME/software/mvapich2-2.3.4/bin/mpirun"
MPIRUN="$HOME/software/openmpi-4.0.5/bin/mpirun"

nodes=1;
for rpn in 1 2 6; do
  let n=$nodes*$rpn

  if   [ $n ==   1 ]; then X=128 Y=128 Z=128
  elif [ $n ==   2 ]; then X=256 Y=128 Z=128
  elif [ $n ==   4 ]; then X=256 Y=256 Z=128
  elif [ $n ==   6 ]; then X=384 Y=256 Z=128
  fi

  echo "${nodes}nodes,${rpn}rankspernode,auto" >> $OUT
  export TEMPI_PLACEMENT_KAHIP=""
  export TEMPI_DATATYPE_AUTO=""
  $MPIRUN -n $n ../../build/bin/bench-halo-exchange 100 $X $Y $Z | tee -a $OUT
  unset TEMPI_PLACEMENT_KAHIP
  unset TEMPI_DATATYPE_AUTO

  # echo "${nodes}nodes,${rpn}rankspernode,oneshot" >> $OUT
  # export TEMPI_PLACEMENT_KAHIP=""
  # export TEMPI_DATATYPE_ONESHOT=""
  # $MPIRUN -n $n ../../build/bin/bench-halo-exchange 100 $X $Y $Z | tee -a $OUT
  # unset TEMPI_PLACEMENT_KAHIP
  # unset TEMPI_DATATYPE_ONESHOT

  # echo "${nodes}nodes,${rpn}rankspernode,device" >> $OUT
  # export TEMPI_PLACEMENT_KAHIP=""
  # export TEMPI_DATATYPE_DEVICE=""
  # $MPIRUN -n $n ../../build/bin/bench-halo-exchange 100 $X $Y $Z | tee -a $OUT
  # unset TEMPI_PLACEMENT_KAHIP
  # unset TEMPI_DATATYPE_DEVICE

#    echo "${nodes}nodes,${rpn}rankspernode,notempi" >> $OUT
#    export TEMPI_DISABLE=""
#    jsrun --smpiargs="-gpu" -n $n -r $rpn -a 1 -g 1 -c 7 -b rs ../../build/bin/bench-halo-exchange 5 $X $Y $Z | tee -a $OUT
#    unset TEMPI_DISABLE
done


