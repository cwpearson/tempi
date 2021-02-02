#!/bin/bash

set -eou pipefail

DIR=$HOME/tempi_results
OUT=$DIR/measure_system.txt

set -x

mkdir -p $DIR

echo "" > $OUT

#MPIRUN="$HOME/software/mvapich2-2.3.4/bin/mpirun -n 2"
MPIRUN="$HOME/software/openmpi-4.0.5/bin/mpirun -n 2"

$MPIRUN ../../build/bin/measure-system | tee -a $OUT


