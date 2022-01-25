host=`hostname`

if [[ "$host" =~ .*vortex.* ]]; then
# CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

    echo "$host" matched vortex
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+

    echo module --force purge
    module --force purge

    echo module load cmake/3.18.0
    module load cmake/3.18.0
    echo module load cuda/10.2.89
    module load cuda/10.2.89
    echo module load gcc/8.3.1
    module load gcc/8.3.1
    echo module load spectrum-mpi/2020.08.19
    module load spectrum-mpi/2020.08.19

    #export MPI_ROOT=$HOME/software/openmpi3.1.4-cuda10.2.89-gcc7.3.1
    #export LD_LIBRARY_PATH=$HOME/software/openmpi3.1.4-cuda10.2.89-gcc7.3.1/lib

    which cmake
    which gcc
    which nvcc
    which mpirun
elif [[ "$host" =~ blake ]]; then
    module load python/3.7.3
    module load openmpi/4.1.0/gcc/7.2.0
    module load cmake/3.19.3
fi

