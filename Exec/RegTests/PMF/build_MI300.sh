#!/bin/bash -l
model=LiDryer
arch=gfx940
nompi=NO

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
        -with-mpi=*|--with-mpi=*)
	    mpi="${i#*=}"
	    shift # past argument=value
	    ;;
        -without-mpi|--without-mpi)
	    nompi=YES
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
export PATH=/opt/cmake-3.24.2/:$PATH
MPI=TRUE

if [ $nompi == "YES" ];
then
    # explicitly disable MPI
    echo "Building without MPI"
    MPI=FALSE
    
elif [ -z ${mpi+x} ]; #check if value is passed as an arg
then
    #attempt to read any ompi version in /opt
    mpi=$(readlink -f /opt/omp*)
    if test -d $mpi; then
	export PATH=$mpi/bin/:$PATH
	export MPI_HOME=$mpi
    else
	echo "Could NOT find MPI Path. Exiting"
	exit 1
    fi
else
    if test -d $mpi/; #check value of passed argument
    then
	export PATH=$mpi/bin/:$PATH
	export MPI_HOME=$mpi
    else
	echo "MPI arg not a real path. Exiting"
	exit 1
    fi
fi

export HIP_PLATFORM=amd
TRACE=FALSE

#Build SUNDIALS (requires internet connection because a clone happens)
make -j 1 USE_ROCTX=$TRACE USE_MPI=$MPI Chemistry_Model=$model TPLrealclean
make -j 32 USE_ROCTX=$TRACE USE_MPI=$MPI Chemistry_Model=$model TPL

#Build PeleC
make -j 1 USE_ROCTX=$TRACE USE_MPI=$MPI Chemistry_Model=$model realclean
make -j 32 USE_ROCTX=$TRACE USE_MPI=$MPI Chemistry_Model=$model

# Copy the executable
if [ $nompi == "YES" ];
then
    mv PeleC3d.hip.TPROF.HIP.ex PeleC3d.hip.TPROF.HIP.ex.$model
else
    mv PeleC3d.hip.TPROF.MPI.HIP.ex PeleC3d.hip.TPROF.MPI.HIP.ex.$model
fi
