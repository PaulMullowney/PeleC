#!/bin/bash
MODEL="${MODEL:-drm19}"
MACHINE="${MACHINE:-smc6219-mi325x}"
for i in 0 1; do
  for j in 1 2 3 4; do
      CFRHS_MULTI_KERNEL=$i CFRHS_MIN_BLOCKS=$j ATOMIC_REDUCTIONS=0 THRUST_REDUCTIONS=0 MODEL=${MODEL} ./run-pelec-${MACHINE}.sh > ${MACHINE}_${MODEL}_lds_reducer_${i}${j}.txt
      CFRHS_MULTI_KERNEL=$i CFRHS_MIN_BLOCKS=$j ATOMIC_REDUCTIONS=0 THRUST_REDUCTIONS=1 MODEL=${MODEL} ./run-pelec-${MACHINE}.sh > ${MACHINE}_${MODEL}_thrust_reducer_${i}${j}.txt
      CFRHS_MULTI_KERNEL=$i CFRHS_MIN_BLOCKS=$j ATOMIC_REDUCTIONS=1 THRUST_REDUCTIONS=0 MODEL=${MODEL} ./run-pelec-${MACHINE}.sh > ${MACHINE}_${MODEL}_atomic_reducer_${i}${j}.txt
  done
done
