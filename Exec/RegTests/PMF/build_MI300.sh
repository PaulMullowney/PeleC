#!/bin/bash -l
model=LiDryer
arch=gfx940

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
export ROCM_PATH=/opt/rocm

if test -d /opt/ompi-5.0.0/; then
    export PATH=/opt/cmake-3.24.2/:/opt/ompi-5.0.0/bin/:$PATH
    export MPI_HOME=/opt/ompi-5.0.0/
elif test -d /opt/ompi-5.0.0-rc12/; then
    export PATH=/opt/cmake-3.24.2/:/opt/ompi-5.0.0-rc12/bin/:$PATH
    export MPI_HOME=/opt/ompi-5.0.0-rc12/
else
    echo "Could NOT find MPI Path. Exiting"
    exit 1
fi

export HIP_PLATFORM=amd
TRACE=FALSE

#Build SUNDIALS (requires internet connection because a clone happens)
make -j 1 USE_ROCTX=$TRACE Chemistry_Model=$model TPLrealclean
make -j 32 USE_ROCTX=$TRACE Chemistry_Model=$model TPL

#Build PeleC
make -j 1 USE_ROCTX=$TRACE Chemistry_Model=$model realclean
make -j 32 USE_ROCTX=$TRACE Chemistry_Model=$model

# Copy the executable
mv PeleC3d.hip.TPROF.MPI.HIP.ex PeleC3d.hip.TPROF.MPI.HIP.ex.$model
