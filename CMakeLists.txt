# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories
# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(tempi LANGUAGES C CXX CUDA VERSION 0.1.0.0)

option(TEMPI_ENABLE_KAHIP "allow KaHIP for placement" ON)
option(TEMPI_ENABLE_METIS "allow metis for placement" OFF)


# Set a log level if none was specified
if(NOT DEFINED TEMPI_OUTPUT_LEVEL)
message(STATUS "TEMPI_OUTPUT_LEVEL not defined, setting to INFO")
set(TEMPI_OUTPUT_LEVEL INFO CACHE STRING "one of SPEW DEBUG INFO WARN ERROR FATAL")
endif()

# Set a log level if none was specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake")
get_git_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
include(CheckCXXCompilerFlag)

macro(tempi_add_output_level_defines target)
  if (TEMPI_OUTPUT_LEVEL MATCHES SPEW)
    target_compile_definitions(${ARGV0} PUBLIC TEMPI_OUTPUT_LEVEL=5)
  elseif (TEMPI_OUTPUT_LEVEL MATCHES DEBUG)
    target_compile_definitions(${ARGV0} PUBLIC TEMPI_OUTPUT_LEVEL=4)
  elseif (TEMPI_OUTPUT_LEVEL MATCHES INFO)
    target_compile_definitions(${ARGV0} PUBLIC TEMPI_OUTPUT_LEVEL=3)
  elseif (TEMPI_OUTPUT_LEVEL MATCHES WARN)
    target_compile_definitions(${ARGV0} PUBLIC TEMPI_OUTPUT_LEVEL=2)
  elseif (TEMPI_OUTPUT_LEVEL MATCHES ERROR)
    target_compile_definitions(${ARGV0} PUBLIC TEMPI_OUTPUT_LEVEL=1)
  elseif (TEMPI_OUTPUT_LEVEL MATCHES FATAL)
    target_compile_definitions(${ARGV0} PUBLIC TEMPI_OUTPUT_LEVEL=0)
  else()
    message(FATAL_ERROR "unrecognized value of TEMPI_OUTPUT_LEVEL: "  ${STENCIL_OUTPUT_LEVEL}  ", must be one of SPEW DEBUG INFO WARN ERROR FATAL")
  endif()
endmacro()

# only static libraries are suported by the device linker?
# so I think we need STATIC with CUDA_SEPARABLE_COMPILATION
add_subdirectory(src)
add_library(tempi SHARED ${TEMPI_SOURCES})
add_library(tempi::tempi ALIAS tempi)
set_property(TARGET tempi PROPERTY CUDA_SEPARABLE_COMPILATION ON)
tempi_add_output_level_defines(tempi)

add_subdirectory(support)
# "this command should be in the source directory root for CTest to find the test file"
enable_testing() 
add_subdirectory(test)
add_subdirectory(bin)

include(ExternalProject)

if(TEMPI_ENABLE_KAHIP)
  message(STATUS "define TEMPI_ENABLE_KAHIP")
  target_compile_definitions(tempi PRIVATE TEMPI_ENABLE_KAHIP)

  ExternalProject_Add(kahip
    GIT_REPOSITORY git@github.com:cwpearson/KaHIP.git
    GIT_TAG "20ff5e3c52c7f8ba9cf89fe2d14f0f5f1f125236" # tempi brach
    #GIT_TAG "b466cdf2ffb3c5c6124f0b212bf808cdcf5a1d7d" # balanced kaffpa
    #GIT_TAG "08c8a02296ba02e9932dc25ba8e76876e18516d2" # balanced process mapping
    GIT_SUBMODULES_RECURSE TRUE
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/kahip-prefix
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )
  add_dependencies(tempi kahip)
  target_include_directories(tempi SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/kahip-prefix/include)
  target_link_libraries(tempi PRIVATE ${CMAKE_BINARY_DIR}/kahip-prefix/lib/libinterface_static.a)
endif(TEMPI_ENABLE_KAHIP)

if(TEMPI_ENABLE_METIS)
  message(STATUS "define TEMPI_ENABLE_METIS")
  target_compile_definitions(tempi PRIVATE TEMPI_ENABLE_METIS)

  # ExternalProject_Add(metis
  #   GIT_REPOSITORY git@github.com:KarypisLab/METIS.git
  #   GIT_TAG "v5.1.1-DistDGL-v0.5"
  #   GIT_SUBMODULES_RECURSE TRUE
  #   INSTALL_DIR=metis-prefix
  #   CMAKE_ARGS
  #     -D CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/metis-prefix
  # )

  # the metis makescript writes out some files before calling cmake
  # pick this revision of metis due to a bug in a newer version (https://github.com/KarypisLab/METIS/issues/20)
  # remove -march=native for Power9 support
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/metis-prefix/lib/libmetis.a)
    ExternalProject_Add(metis
      GIT_REPOSITORY git@github.com:KarypisLab/METIS.git
      GIT_TAG "18f391689433fabbd76721cfdfdd978b4197d089" # july 25
      GIT_SUBMODULES_RECURSE TRUE
      PATCH_COMMAND sed -i /-march=native/d ${CMAKE_BINARY_DIR}/metis-prefix/src/metis/GKlib/GKlibSystem.cmake
      CONFIGURE_COMMAND make config prefix=${CMAKE_BINARY_DIR}/metis-prefix cc=${CMAKE_C_COMPILER}
      BUILD_COMMAND unset MAKEFLAGS && unset MAKELEVEL && make
      BUILD_IN_SOURCE TRUE
      INSTALL_COMMAND unset MAKEFLAGS && unset MAKELEVEL && make install
      INSTALL_DIR=metis-prefix
      CMAKE_ARGS
        -D CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/metis-prefix
    )
    add_dependencies(tempi metis)
  else()
    message(STATUS "libmetis.a found, not rebuilding. delete metis-prefix and re-run cmake to rebuild")
  endif()
  target_include_directories(tempi SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/metis-prefix/include)
  target_link_libraries(tempi PRIVATE ${CMAKE_BINARY_DIR}/metis-prefix/lib/libmetis.a)
endif(TEMPI_ENABLE_METIS)


set_property(TARGET tempi PROPERTY CXX_STANDARD 17)
set_property(TARGET tempi PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET tempi PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tempi PROPERTY CUDA_STANDARD 17)
set_property(TARGET tempi PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET tempi PROPERTY ENABLE_EXPORTS ON)

target_compile_options(tempi PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
 -Wall
 -Wextra
 -Wcast-align;
 -Wdisabled-optimization;
 -Wformat=2;
 -Winit-self;
 -Wlogical-op;
 -Wmissing-include-dirs;
 -Woverloaded-virtual;
 -Wpointer-arith;
 -Wshadow;
 -Wstrict-aliasing;
 -Wswitch-enum;
 -Wvla;
 >
)

target_compile_options(tempi PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
--Wno-deprecated-gpu-targets;
--expt-extended-lambda;
-Xcompiler=-Wall;
-Xcompiler=-Wextra;
-Xcompiler=-Wcast-align;
-Xcompiler=-Wdisabled-optimization;
-Xcompiler=-Wformat=2;
-Xcompiler=-Winit-self;
-Xcompiler=-Wlogical-op;
-Xcompiler=-Wmissing-include-dirs;
-Xcompiler=-Woverloaded-virtual;
-Xcompiler=-Wpointer-arith;
-Xcompiler=-Wshadow;
-Xcompiler=-Wstrict-aliasing;
-Xcompiler=-Wswitch-enum;
-Xcompiler=-Wvla;
-Xptxas=-v;
>
)

check_cxx_compiler_flag(-march=native CXX_HAS_MARCH)
check_cxx_compiler_flag(-mcpu=native CXX_HAS_MCPU)
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(
    tempi
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
     -G
     -Xcompiler=-fno-omit-frame-pointer
    >
    $<$<COMPILE_LANGUAGE:CXX>:
     -fno-omit-frame-pointer
     -g
    >
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(
    tempi
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
  # always apply mtune
  target_compile_options(
    tempi
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mtune=native>
    $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
  )
  # prefer march=native, but fallback to mcpu=native if needed
  if (CXX_HAS_MARCH)
    message(STATUS "-march=native detected")
    target_compile_options(
      tempi
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
  elseif (CXX_HAS_MCPU)
    message(STATUS "-mcpu=native instead of -march=native")
    target_compile_options(
      tempi
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mcpu=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>
    )
  endif()
endif()


if (USE_CUDA_GRAPH)
  message(STATUS "USE_CUDA_GRAPH=ON, compiling with -DTEMPI_USE_CUDA_GRAPH")
  target_compile_definitions(tempi PRIVATE -DTEMPI_USE_CUDA_GRAPH)
endif()

## Link against required CUDA libraries, even without nvcc
target_link_libraries(tempi PUBLIC CUDA::cudart)
set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
find_library(NVTOOLSEXT_LIB nvToolsExt
            HINTS
            "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib"
            "${CUDA_TOOLKIT_ROOT_DIR}"
            )
message(STATUS "nvToolsExt found: ${NVTOOLSEXT_LIB}")
find_library(NVIDIAML_LIB nvidia-ml
            HINTS
            "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib"
            "${CUDA_TOOLKIT_ROOT_DIR}"
            )
message(STATUS "nvidia-ml found: ${NVIDIAML_LIB}")
target_link_libraries(tempi PUBLIC ${NVTOOLSEXT_LIB})


## find an MPI header to include, but we don't need to link against MPI.
target_include_directories(tempi SYSTEM PRIVATE ${MPI_CXX_INCLUDE_DIRS})

## Add include directories
target_include_directories(tempi SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(tempi PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)





