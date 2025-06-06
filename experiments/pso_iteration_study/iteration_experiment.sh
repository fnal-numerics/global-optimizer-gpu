#!/bin/bash
module load cuda/12.1
echo "Compiling CUDA code.."
#crun.cuda nvcc -o pso parallel_pso.cu
ITER=10000
#N=335544320
N=131072
for FUNC in 1 2 3; do
    PSO=0
    while [ $PSO -le 10 ]; do
        {
            echo $FUNC  # Function selector: 1: Rosenbrock, 2: Rastrigin, 3: Ackley
            echo n      # Do not save trajectory
            echo n      # Exit interactive loop
        } | ./pso -5.12 5.12 "$ITER" "$PSO" 1 "$N" "1e-6"
        PSO=$((PSO + 1))
    done
done
