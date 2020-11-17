set -eou pipefail

MPIRUN="$HOME/software/openmpi-4.0.5/bin/mpirun -n 1"
#MPIRUN="$HOME/software/mvapich2-2.3.4/bin/mpirun -n 1"

DIR=$HOME/sync_work/tempi_results/deneb
OUT=$DIR/bench_type_commit.csv

set -x

mkdir -p $DIR

echo "" > $OUT


echo "tempi" >> $OUT
unset TEMPI_DISABLE
$MPIRUN ../../build/bin/bench-type-commit | tee -a $OUT

echo "notempi" >> $OUT
export TEMPI_DISABLE=""
$MPIRUN ../../build/bin/bench-type-commit | tee -a $OUT
unset TEMPI_DISABLE


