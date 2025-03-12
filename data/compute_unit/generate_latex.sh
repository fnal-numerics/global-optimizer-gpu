#!/usr/bin/env bash

TOTAL_COMPUTE=$1

OUTPUT_DIR="./trajectories"

echo "\begin{table}[ht]"
echo "\centering"
echo "\begin{tabular}{ccccccc}"
echo "\hline"
echo "Iter & NumOpt & Time (ms) & Error & Coordinates & ThreadIndex \\\\"
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

