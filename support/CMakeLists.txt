add_library(tempi_support OBJECT csr.cpp squaremat.cpp type.cpp)
target_include_directories(tempi_support PRIVATE ${MPI_CXX_INCLUDE_DIRS})