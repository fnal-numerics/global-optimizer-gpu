module load cuda/12.1
echo "Compiling CUDA code.."
#crun.cuda nvcc -o pso parallel_pso.cu
#N=335544320

#N=131072   # 2^17 for 10d
#N=1048576  # 2^19 for 32d
#N=8388608  # 2^23
#N=16777216 # 2^24 
#PSO=(0 1 2 3 4 5 10 30) # 100 300 1000)
#PSO=(100 300 1000 3000 10000)
PSO=(100000) # 10000 100000 1000000)
#PSO=(16 128 1024 8192 65536 131072)
TOTAL_COMPUTE=$1
DIM=128
NUM_OPT=2
C=1
while [ $NUM_OPT -le $TOTAL_COMPUTE ]; do
    ITER=$((TOTAL_COMPUTE / NUM_OPT))
#for i in {1..100}
#do    
    #echo -e "\n\n\t\t=== Run $i ===="
    #for ITER in "${PSO[@]}"; do
      echo "=== Running Zeus with $ITER particles ==="
      for FUNC in 1; do # 2 3; do
        if [ $FUNC -eq 1 ]; then
          THRESHOLD="1e-18"  # rosenbrock
        elif [ $FUNC -eq 2 ]; then
          THRESHOLD="5e-04" # rastrigin
        elif [ $FUNC -eq 3 ]; then
          THRESHOLD="1.5"  # ackley
        else
          THRESHOLD="1e-4"
        fi
        {
             echo $FUNC  # Function selector: 1: Rosenbrock, 2: Rastrigin, 3: Ackley
             echo n      # Do not save trajectory
             echo n      # Exit interactive loop
         } | crun.cuda ./100dzeus -5.12 5.12 "$ITER" 5 "$NUM_OPT" "$NUM_OPT" "$THRESHOLD" "$C" "$ITER"
        NUM_OPT=$((NUM_OPT * 2))
        #C=$((C+1))
        done
     #done
done
