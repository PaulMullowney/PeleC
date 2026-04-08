#!/bin/bash

RANKS="${RANKS:-8}"
CFRHS_MULTI_KERNEL="${CFRHS_MULTI_KERNEL:-0}"
CFRHS_MIN_BLOCKS="${CFRHS_MIN_BLOCKS:-2}"
ATOMIC_REDUCTIONS="${ATOMIC_REDUCTIONS:-0}"
THRUST_REDUCTIONS="${THRUST_REDUCTIONS:-0}"
MAX_STEP="${MAX_STEP:-25}"
ARENA_SIZE="${ARENA_SIZE:-150000000000}"
MODEL="${MODEL:-drm19}"
N_CELL="${N_CELL:-256 256 256}"

module load rocm/7.2.0 openmpi/5.0.10-ucx-1.20.0-ucc-1.7.0-gnu
module load cmake/3.31.7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFFINITY_SH="$SCRIPT_DIR/affinity.sh"
LAUNCHER="${LAUNCHER:-mpirun -np $RANKS --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 --mca pml ucx --mca osc ucx --mca opal_rocm_support 1 -x HIP_MEM_POOL_SUPPORT=1 -x HSA_ENABLE_SDMA=0  -x UCX_TLS=sm,self,rocm --map-by ppr:4:numa:PE=16 --bind-to core $AFFINITY_SH}"

COMMON_ARGS="geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=$N_CELL max_step=$MAX_STEP amrex.the_arena_init_size=$ARENA_SIZE amrex.use_gpu_aware_mpi=1"

ODE_ARGS="ode.atomic_reductions=$ATOMIC_REDUCTIONS ode.thrust_reductions=$THRUST_REDUCTIONS ode.cfrhs_multi_kernel=$CFRHS_MULTI_KERNEL ode.cfrhs_min_blocks=$CFRHS_MIN_BLOCKS"

EXEC="${EXEC:-./PeleC3d.hip.TPROF.MPI.HIP.ex.${MODEL}}"
if [[ $MODEL == "drm19" ]]
then
    ARGS="pmf-drm19-cvode.inp $COMMON_ARGS $ODE_ARGS"
    $LAUNCHER $EXEC ${ARGS}
    

elif [[ $MODEL == "dodecane_lu" ]]
then
    # ParmParse overrides on top of pmf-dodecane.inp (parameters not set by COMMON_ARGS / prob.standoff).
    DODECANE_LU_ARGS="stop_time=6 geometry.is_periodic=1 1 0 geometry.coord_sys=0 pelec.cfl=0.05 pelec.init_shrink=0.1 pelec.change_max=1.1 pelec.dt_cutoff=5.e-20 pelec.sum_interval=1 pelec.v=1 amr.v=1 amr.max_level=1 amr.ref_ratio=2 2 2 2 amr.regrid_int=2 2 2 2 amr.blocking_factor=32 amr.max_grid_size=64 amr.n_error_buf=2 2 2 2 amr.checkpoint_files_output=0 amr.check_int=500 amr.plot_files_output=0 amr.plot_int=500 prob.pamb=1013250.0 prob.phi_in=-0.5 prob.pertmag=0.01 prob.pmf_datafile=PMF_NC12H26_1bar_300K_DodecaneLu.dat amr.loadbalance_with_workestimates=1 tagging.max_ftracerr_lev=4 tagging.ftracerr=150.e-6 pelec.do_hydro=1 pelec.do_react=1 pelec.chem_integrator=ReactorCvode cvode.solve_type=GMRES ode.rtol=1e-4 ode.atol=1e-5 pelec.diffuse_temp=1 pelec.diffuse_enth=1 pelec.diffuse_spec=1 pelec.diffuse_vel=1 pelec.sdc_iters=2 pelec.flame_trac_name=HO2 amrex.signal_handling=0 amrex.abort_on_out_of_gpu_memory=1 amrex.the_arena_is_managed=0 pelec.use_typ_vals_chem=1 pelec.typical_rhoY_val_min=1e-6 pelec.do_mol=0"
    ARGS="pmf-dodecane.inp $COMMON_ARGS $ODE_ARGS prob.standoff=-1.0 $DODECANE_LU_ARGS"
    $LAUNCHER $EXEC ${ARGS}

else
    echo "run-pelec.sh: set MODEL to one of: drm19, dodecane_lu" >&2
    exit 1
fi
