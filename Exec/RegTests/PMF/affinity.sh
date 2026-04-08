#!/bin/bash
if [[ -n ${OMPI_COMM_WORLD_LOCAL_RANK+z} ]]; then
    export MPI_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
    export MPI_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}
elif [[ -n "${SLURM_LOCALID+z}" ]]; then
    export MPI_RANK=${SLURM_LOCALID}
    export MPI_SIZE=${SLURM_NTASKS}
else
    export MPI_RANK=0
    export MPI_SIZE=1
fi

dev=$(( MPI_RANK % MPI_SIZE ))
if [[ "$MPI_SIZE" == "64" ]]; then
    #DEVICES_LIST=(1 3 2 0 5 7 6 4)
    DEVICES_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63)
elif [[ "$MPI_SIZE" == "32" ]]; then
    #DEVICES_LIST=(1 3 2 0 5 7 6 4)
    DEVICES_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
    #DEVICES_LIST=(4 5 6 7 12 13 14 15 8 9 10 11 0 1 2 3 20 21 22 23 28 29 30 31 24 25 26 27 16 17 18 19)
elif [[ "$MPI_SIZE" == "16" ]]; then
    #DEVICES_LIST=(1 3 2 0 5 7 6 4)
    DEVICES_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    #DEVICES_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
elif [[ "$MPI_SIZE" == "8" ]]; then
    DEVICES_LIST=(0 1 2 3 4 5 6 7)
elif [[ "$MPI_SIZE" == "4" ]]; then
    DEVICES_LIST=(1 3 5 7)
elif [[ "$MPI_SIZE" == "2" ]]; then
    DEVICES_LIST=(1 5)
    DEVICES_LIST=(1 3)
else
    DEVICES_LIST=(1)
fi
export ROCR_VISIBLE_DEVICES=$dev #${DEVICES_LIST[$dev]}
#export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
$@
