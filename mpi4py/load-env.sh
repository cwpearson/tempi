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
elif [[ "$host" =~ blake ]]; then

    module load python/3.7.3
    export PYTHONHTTPSVERIFY=0

    module load openmpi/4.1.0/gcc/7.2.0

elif [[ "$host" =~ vortex ]]; then

    echo "$host" matched vortex

    # needed a newer gcc to install poetry
    echo module load gcc/7.3.1
    module load gcc/7.3.1

    # newer OpenSSL for python crypotgraphy install
    export LIB=$HOME/software/openssl-1.1.1m-ppc64le/lib
    export INCLUDE=$HOME/software/openssl-1.1.1m-ppc64le/include

elif [[ "$host" =~ ascicgpu ]]; then

    echo "$host" matched ascicgpu

fi