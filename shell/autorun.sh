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
echo "Compiling CUDA code"
crun.cuda nvcc -o exe main.cu

echo "Select function to optimize:"
echo " 1) Rosenbrock"
echo " 2) Rastrigin"
echo " 3) Ackley"
echo " 4) GoldsteinPrice"
echo " 5) Eggholder"
echo " 6) Himmelblau"
echo " 7) Custom"
read -p "Enter your choice: " FUNC_CHOICE

# map choice to lowercase name
case $FUNC_CHOICE in
    1) FUNC_NAME="rosenbrock";;
    2) FUNC_NAME="rastrigin";;
    3) FUNC_NAME="ackley";;
    4) FUNC_NAME="goldsteinprice";;
    5) FUNC_NAME="eggholder";;
    6) FUNC_NAME="himmelblau";;
    7) FUNC_NAME="custom";;
    *) echo "Invalid choice"; exit 1;;
esac

OUTPUT_DIR="../data/${FUNC_NAME}/${TOTAL_COMPUTE}"
mkdir -p "$OUTPUT_DIR"

# Initialize TSV file with header
TSV_FILE="${OUTPUT_DIR}/results.tsv"
echo -e "Iter\tNumOpt\tTime (ms)\tError\tx[0]\tx[1]\tThreadIndex" > "$TSV_FILE"

# generate powers of two for max_iter from 1 up to TOTAL_COMPUTE
#   for each max_iter = 1, 2, 4, 8, ..., calculate num_opt = TOTAL_COMPUTE / max_iter
#     then run the program with those arguments
#     saving output
NUM_OPT=1
while [ $NUM_OPT -le $TOTAL_COMPUTE ]; do
    ITER=$((TOTAL_COMPUTE / NUM_OPT))
    echo "Running with max_iter=$ITER and num_opt=$NUM_OPT"
    {
        echo "$FUNC_CHOICE"  # Send the user's function choice
        echo n             # Exit the interactive loop in the program
    } | ./exe -5.0 5.0 "$ITER" "$NUM_OPT" &> ./"$OUTPUT_DIR"/"${ITER}it_${NUM_OPT}.txt"
    
    NUM_OPT=$((NUM_OPT * 2))
done

echo "\begin{table}[ht]"
echo "\centering"
echo "\begin{tabular}{ccccccc}"
echo "\hline"
echo "Iter & NumOpt & Time (ms) & Error & x[0] & x[1] & ThreadIndex \\\\"
echo "\hline"

# gather all .txt files in an array
files=( "$OUTPUT_DIR"/*.txt )

# for each file, parse out the iteration value and store in a map/dict
declare -A fileIter
for f in "${files[@]}"; do
  BASENAME=$(basename "$f")
  ITER=$(echo "$BASENAME" | sed -E 's/^([0-9]+)it_.*/\1/')
  fileIter["$f"]=$ITER
done

# sort files by iteration in descending order
sortedFiles=($(for f in "${files[@]}"; do
  echo "${fileIter[$f]} $f"
done | sort -k1,1nr | awk '{print $2}'))

# process files in descending iteration order
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
echo "\caption{Results for ${FUNC_NAME} total compute of $TOTAL_COMPUTE.}"
echo "\label{tab:${FUNC_NAME}-${TOTAL_COMPUTE}-results}"
echo "\end{table}"

