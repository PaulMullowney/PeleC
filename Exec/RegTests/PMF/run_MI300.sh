#!/bin/bash -l

model=LiDryer
ranks=1
cfrhs_multi_kernel=0
cfrhs_min_blocks=2
nompi=NO

for i in "$@"; do
    case "$1" in
        -model=*|--model=*)
            model="${i#*=}"
            shift # past argument=value
            ;;
        -arena_size=*|--arena_size=*)
            arena_size="${i#*=}"
            shift # past argument=value
            ;;
        -max_step=*|--max_step=*)
            max_step="${i#*=}"
            shift # past argument=value
            ;;
        -ranks=*|--ranks=*)
            ranks="${i#*=}"
            shift # past argument=value
            ;;
        -cfrhs_multi_kernel=*|--cfrhs_multi_kernel=*)
            cfrhs_multi_kernel="${i#*=}"
            shift # past argument=value
            ;;
        -cfrhs_min_blocks=*|--cfrhs_min_blocks=*)
            cfrhs_min_blocks="${i#*=}"
            shift # past argument=value
            ;;
        -with-mpi=*|--with-mpi=*)
	    mpi="${i#*=}"
	    shift # past argument=value
	    ;;
        -build=*|--build=*)
 	    build="${i#*=}"
	    shift # past argument=value
	    ;;
        -without-mpi|--without-mpi)
	    nompi=YES
	    shift # past argument=value
	    ;;
	-fast|--fast)
            fast=YES
            shift # past argument=value
            ;;
        --)
            shift
            break
            ;;
    esac
done

if [ $nompi == "YES" ];
then
    # explicitly disable MPI
    echo "Running without MPI with 1 rank"
    ranks=1
    
elif [ -z ${mpi+x} ]; #check if value is passed as an arg
then
    #attempt to read any ompi version in /opt
    mpi=$(readlink -f /opt/omp*)
    if test -d $mpi; then
	export PATH=$mpi/bin/:$PATH
    else
	echo "Could NOT find MPI Path. Exiting"
	exit 1
    fi
else
    if test -d $mpi/; #check value of passed argument
    then
	export PATH=$mpi/bin/:$PATH
    else
	echo "MPI arg not a real path. Exiting"
	exit 1
    fi
fi

echo $model, $ranks

if [[ $model == "LiDryer" ]]
then
    if [[ $max_step ]]
    then
	MAX_STEP=$max_step
    else
	MAX_STEP=25
    fi

    if [[ $arena_size ]]
    then
	ARENA_SIZE=$arena_size
    else
	ARENA_SIZE=48000000000
    fi
    ARGS="pmf-lidryer-cvode.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks"
    if [ $nompi == "YES" ];
    then
	./PeleC3d.hip.TPROF.HIP.ex.$model ${ARGS}
    else
	mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}
    fi

elif [[ $model == "drm19" ]]
then
    #  pelec.init_shrink=1.0 pelec.change_max=1.0 
    if [[ $max_step ]]
    then
	MAX_STEP=$max_step
    else
	MAX_STEP=25
    fi
    if [[ $arena_size ]]
    then
	ARENA_SIZE=$arena_size
    else
	ARENA_SIZE=48000000000
    fi
    ARGS="pmf-drm19-cvode.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks"
    if [ $nompi == "YES" ];
    then
	./PeleC3d.hip.TPROF.HIP.ex.$model ${ARGS}
    else
	mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}
    fi

elif [[ $model == "dodecane_lu" ]]
then

    if [[ $max_step ]]
    then
	MAX_STEP=$max_step
    else
	MAX_STEP=10
    fi

    if [[ $arena_size ]]
    then
	ARENA_SIZE=$arena_size
    else
	ARENA_SIZE=60000000000
    fi

    if [[ $fast ]]
    then
	ARGS="pmf-dodecane-cvode.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks prob.standoff=-1.0 amrex.max_gpu_streams=1"
    else
	ARGS="pmf-dodecane-cvode.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks pelec.init_shrink=1.0 pelec.change_max=1.0"
    fi
    if [ $nompi == "YES" ];
    then
	./PeleC3d.hip.TPROF.HIP.ex.$model.$build ${ARGS}
    else
	mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}
    fi

elif [[ $model == "isooctane_lu" ]]
then

    if [[ $max_step ]]
    then
	MAX_STEP=$max_step
    else
	MAX_STEP=10
    fi

    if [[ $arena_size ]]
    then
	ARENA_SIZE=$arena_size
    else
	ARENA_SIZE=60000000000
    fi

    ARGS="pmf-isooctane.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=2.5 2.5 6.0 amr.n_cell=64 64 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks"
    if [ $nompi == "YES" ];
    then
	./PeleC3d.hip.TPROF.HIP.ex.$model ${ARGS}
    else
	mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}
    fi

fi
