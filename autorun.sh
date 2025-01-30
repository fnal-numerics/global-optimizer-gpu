#!/usr/bin/env bash

# Compile and run a CUDA program with varying iteration/optimization parameters
# Saving the output in a directory named <TOTAL_COMPUTE>/<max_iter>it_<num_opt>.txt
# Usage:
#   ./auto_run.sh <TOTAL_COMPUTE>
# Example:
#   ./auto_run.sh 1024

# 1) Pass the total compute as the first argument
TOTAL_COMPUTE=$1

module load cuda/12.1
# Compile the CUDA program
crun.cuda nvcc -o exe main.cu

# 3) Create a directory named after the total compute if it doesn't exist
mkdir -p "$TOTAL_COMPUTE"

# 4) Generate powers of two for max_iter from 1 up to TOTAL_COMPUTE
#    For each max_iter = 1, 2, 4, 8, ..., calculate num_opt = TOTAL_COMPUTE / max_iter
#    Then run the program with those arguments, saving output.
ITER=1
while [ $ITER -le $TOTAL_COMPUTE ]; do
    NUM_OPT=$((TOTAL_COMPUTE / ITER))
    echo "Running with max_iter=$ITER and num_opt=$NUM_OPT"
    ./exe -5.0 5.0 "$ITER" "$NUM_OPT" \
      &> ./"$TOTAL_COMPUTE"/"${ITER}it_${NUM_OPT}.txt"
    ITER=$((ITER * 2))
done

