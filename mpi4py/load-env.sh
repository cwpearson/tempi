host=`hostname`


if [[ "$NERSC_HOST" =~ perlmutter ]]; then

    echo $NERCS_HOST matched perlmutter

    echo module load cray-python/3.9.4.2
    module load cray-python/3.9.4.2

    echo module load nvidia/21.9
    module load nvidia/21.9
    export MPICC=`readlink -f cc-wrapper.sh`
    export MPICXX=CC

    # echo module load gcc/9.3.0
    # module load gcc/9.3.0
    # export MPICC=gcc
    # export MPICXX=g++
fi