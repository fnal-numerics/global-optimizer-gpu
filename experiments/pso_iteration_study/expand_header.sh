# Detect how many columns the second line has
num_cols=$(head -n2 5d_results.tsv | tail -n1 | awk -F'\t' '{print NF}')

# Number of fixed columns (e.g. 6 before coordinates)
fixed_cols=6

# How many coordinate fields there are
num_coords=$((num_cols - fixed_cols))

# Create a corrected header
(
  echo -ne "fun\titer\tpso_iter\ttime\terror\tfnval"
  for i in $(seq 1 $num_coords); do
    echo -ne "\tcoord_$i"
  done
  echo
  tail -n +2 5d_results.tsv
) > fixed_5d_results.tsv

mv fixed_5d_results.tsv 5d_results.tsv

