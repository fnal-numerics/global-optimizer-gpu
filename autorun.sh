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

echo "\begin{table}[ht]"
echo "\centering"
echo "\begin{tabular}{ccccccc}"
echo "\hline"
echo "Iter & NumOpt & Time (ms) & Error & x[0] & x[1] & ThreadIndex \\\\"
echo "\hline"

# 1) Gather all .txt files in an array
files=( "$TOTAL_COMPUTE"/*.txt )

# 2) For each file, parse out the iteration value and store in a map/dict
declare -A fileIter
for f in "${files[@]}"; do
  BASENAME=$(basename "$f")
  ITER=$(echo "$BASENAME" | sed -E 's/^([0-9]+)it_.*/\1/')
  fileIter["$f"]=$ITER
done

# 3) Sort files by iteration in descending order
sortedFiles=($(for f in "${files[@]}"; do
  echo "${fileIter[$f]} $f"
done | sort -k1,1nr | awk '{print $2}'))

# 4) Process files in descending iteration order
for f in "${sortedFiles[@]}"; do
  BASENAME=$(basename "$f")
  ITER=$(echo "$BASENAME" | sed -E 's/^([0-9]+)it_.*/\1/')
  NUMOPT=$(echo "$BASENAME" | sed -E 's/.*it_([0-9]+)\.txt$/\1/')

  TIME=$(grep -m1 "^Optimization Kernel execution time" "$f" \
    | sed -E 's/.*=\s*([0-9\.]+)\s*ms.*$/\1/')

  ERROR=$(grep -m1 "^Global Minima:" "$f" \
    | sed -E 's/^Global Minima:\s*([0-9.\-]+).*/\1/')

  THREADIDX=$(grep -m1 "^Global Minima Index:" "$f" \
    | sed -E 's/^Global Minima Index:\s*([0-9]+).*/\1/')

  X0=$(grep -m1 "^x\[0\]" "$f" \
    | sed -E 's/^x\[0\]\s*=\s*([0-9.\-]+)/\1/')

  X1=$(grep -m1 "^x\[1\]" "$f" \
    | sed -E 's/^x\[1\]\s*=\s*([0-9.\-]+)/\1/')

  echo "$ITER & $NUMOPT & $TIME & $ERROR & $X0 & $X1 & $THREADIDX \\\\"
done

echo "\hline"
echo "\end{tabular}"
echo "\caption{Results for total compute of $TOTAL_COMPUTE.}"
echo "\label{tab:${TOTAL_COMPUTE}results}"
echo "\end{table}"

