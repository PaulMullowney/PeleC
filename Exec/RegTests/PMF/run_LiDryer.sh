#!/bin/bash -l
arena_size=48000000000

ARGS="example.inp geometry.prob_lo=0. 0. 1. geometry.prob_hi=5.0 5.0 6.0 amr.n_cell=128 128 128 amr.max_level=1 tagging.ftracerr=1.e-4 prob.pmf_datafile=LiDryer_H2_p1_phi0_4000tu0300.dat max_step=25 amr.plot_files_output=0 amr.plot_int=10 amr.checkpoint_files_output=0 amrex.abort_on_out_of_gpu_memory=1 pelec.init_shrink=1.0 pelec.change_max=1.0 amrex.the_arena_is_managed=0 pelec.chem_integrator=ReactorCvode cvode.solve_type=GMRES amr.blocking_factor=32 amr.max_grid_size=64 pelec.use_typ_vals_chem=1 ode.rtol=1e-4 ode.atol=1e-5 pelec.typical_rhoY_val_min=1e-6 pelec.do_mol=0 amrex.the_arena_init_size=$arena_size"

mpirun -n 1 ./PeleC3d.hip.TPROF.MPI.HIP.ex ${ARGS}
