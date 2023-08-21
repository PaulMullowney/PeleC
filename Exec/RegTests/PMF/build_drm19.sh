#!/bin/bash -l
export AMD_ARCH=gfx940
export ROCM_PATH=/opt/rocm
export PATH=/opt/cmake-3.24.2/:/opt/ompi-5.0.0rc12/bin/:$PATH
export MPI_HOME=/opt/ompi-5.0.0rc12/
export HIP_PLATFORM=amd
MODEL=drm19

#Build SUNDIALS (requires internet connection because a clone happens)
make -j 1 Chemistry_Model=$MODEL TPLrealclean
make -j 32 Chemistry_Model=$MODEL TPL

#Build PeleC
make -j 1  Chemistry_Model=$MODEL realclean
make -j 32 Chemistry_Model=$MODEL
