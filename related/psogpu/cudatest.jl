using CUDA

# Check CUDA version and device info
println("CUDA version: ", CUDA.version())
println("Device: ", CUDA.device())

# Allocate a CUDA array and perform basic operations
x = CUDA.fill(1.0f0, 1024)  # Allocate a 1024-element float array on the GPU
println("CUDA Array: ", x)
