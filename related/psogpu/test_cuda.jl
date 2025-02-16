using CUDA

# Get the active device
device = CUDA.device()

# Print device information
println("CUDA Device: ", device)
println("Device Name: ", CUDA.name(device))

# Ensure the device is a GPU and get the compute capability
if CUDA.has_cuda()
    println("Compute Capability: ", CUDA.capability(device))
else
    println("No CUDA-capable device found.")
end

# Print total memory on the device
println("Total Memory (bytes): ", CUDA.totalmem(device))
