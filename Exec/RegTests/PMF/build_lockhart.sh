#!/bin/bash -l

model=LiDryer
arch=gfx90a

for i in "$@"; do
    case "$1" in
        -model=*|--model=*)
            model="${i#*=}"
            shift # past argument=value
            ;;
        -arch=*|--arch=*)
            arch="${i#*=}"
            shift # past argument=value
            ;;
        --)
            shift
            break
            ;;
    esac
done

export AMD_ARCH=$arch
export PATH=/home/pmullown/software/bin/:$PATH
module purge
module load PrgEnv-cray
module load xpmem
module unload cray-libsci
module load cray-libsci/22.10.1.2
module load craype-x86-rome craype-accel-amd-gfx90a amd/5.4.3 cray-mpich
export MPI_HOME=/opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/
export HIP_PLATFORM=amd
TRACE=FALSE

#Build SUNDIALS (requires internet connection because a clone happens)
make -j 1 USE_ROCTX=$TRACE Chemistry_Model=$model TPLrealclean
make -j 32 USE_ROCTX=$TRACE Chemistry_Model=$model TPL

#Build PeleC
make -j 1 USE_ROCTX=$TRACE Chemistry_Model=$model realclean
make -j 32 USE_ROCTX=$TRACE Chemistry_Model=$model

# Copy the executable
mv PeleC3d.hip.x86-rome.TPROF.MPI.HIP.ex PeleC3d.hip.x86-rome.TPROF.MPI.HIP.ex.$model
