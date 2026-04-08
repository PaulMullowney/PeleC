#!/bin/bash -l

MODEL="${MODEL:-drm19}"

#module use /mnt/share/Modules/install/nvhpc/12.9/modulefiles
#module load nvhpc-hpcx/25.5
module purge
module load cuda/13.2
source ~/paul-openmpi/init-openmpi-cuda.rc

TRACE=FALSE

#Build SUNDIALS (requires internet connection because a clone happens)
make -j 1 USE_MPI=TRUE USE_CUDA=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL TPLrealclean
make -j 32 USE_MPI=TRUE USE_CUDA=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL TPL

#Build PeleC
make -j 1 USE_MPI=TRUE USE_CUDA=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL realclean
make -j 32 USE_MPI=TRUE USE_CUDA=TRUE TINY_PROFILE=TRUE Chemistry_Model=$MODEL

# Copy the executable
mv PeleC3d.gnu.TPROF.MPI.CUDA.ex PeleC3d.gnu.TPROF.MPI.CUDA.ex.$MODEL
