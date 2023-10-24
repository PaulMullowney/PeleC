#!/bin/bash -l

export PATH=/opt/ompi-5.0.0rc12/bin/:$PATH
model=LiDryer
ranks=1
cfrhs_multi_kernel=0
cfrhs_min_blocks=2

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
    mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}

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
    mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}

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
	ARGS="pmf-dodecane-cvode.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks prob.standoff=-1.0"
    else
	ARGS="pmf-dodecane-cvode.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE ode.cfrhs_multi_kernel=$cfrhs_multi_kernel ode.cfrhs_min_blocks=$cfrhs_min_blocks pelec.init_shrink=1.0 pelec.change_max=1.0"
    fi
    mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}

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
    mpirun -n $ranks ./PeleC3d.hip.TPROF.MPI.HIP.ex.$model ${ARGS}

fi
