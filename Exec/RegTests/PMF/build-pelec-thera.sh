#!/bin/bash -l

MODEL="${MODEL:-drm19}"
ARCH="${ARCH:-gfx950}"

module load rocm/7.2.0 openmpi/5.0.9-ucc1.7.0-ucx1.20.0-rocm7.2.0
module load cmake/3.31.7

export PATH=$THERA_OMPI_DIR/bin/:$PATH
export MPI_HOME=$THERA_OMPI_DIR

export AMD_ARCH=$ARCH

export HIP_PLATFORM=amd
TRACE=TRUE

#Build SUNDIALS (requires internet connection because a clone happens)
make -j 1 USE_ROCTX=$TRACE USE_MPI=TRUE USE_HIP=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL TPLrealclean
make -j 32 USE_ROCTX=$TRACE USE_MPI=TRUE USE_HIP=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL TPL

#Build PeleC
make -j 1 USE_ROCTX=$TRACE USE_MPI=TRUE USE_HIP=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL realclean
make -j 32 USE_ROCTX=$TRACE USE_MPI=TRUE USE_HIP=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL

# Copy the executable
mv PeleC3d.hip.TPROF.MPI.HIP.ex PeleC3d.hip.TPROF.MPI.HIP.ex.$MODEL
