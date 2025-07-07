#!/bin/bash
module load cuda/12.1
echo "Compiling CUDA code.."
#crun.cuda nvcc -o pso parallel_pso.cu
ITER=10000
#N=335544320
N=1000000
for i in {1..100}
do
    echo -e "\n\n\t\t=== Run $i ===="
    for FUNC in 1 2; do # only do rosenbrock and rastrigin
        PSO=0
        #if [ $FUNC -eq 1 ]; then
        #  THRESHOLD="6e-4"  # rosenbrock
        #elif [ $FUNC -eq 2 ]; then
        #  THRESHOLD="5e-04" # rastrigin
        #fi
        while [ $PSO -le 10 ]; do
            {
                echo $FUNC  # Function selector: 1: Rosenbrock, 2: Rastrigin, 3: Ackley
                echo n      # Do not save trajectory
                echo n      # Exit interactive loop
             } | crun.cuda ./10dzeus -5.12 5.12 "$ITER" "$PSO" 100 "$N" "1e-6" "$((($i*$PSO) + $PSO))" "$i" 
             PSO=$((PSO + 1))
        done
    done
done
