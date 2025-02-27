#!/bin/bash

# Usage check
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <lower_bound> <upper_bound> <num_iterations> <num_optimizations>"
    exit 1
fi

LOWER=$1
UPPER=$2
ITER=$3
OPTIM=$4

# clean and create build folder
rm -rf build && mkdir build && cd build

# run CMake and build
crun cmake ..
crun make

# run tests via CTest
crun ctest --output-on-failure

./cuda_app $LOWER $UPPER $ITERATIONS $OPTIMIZATIONS

