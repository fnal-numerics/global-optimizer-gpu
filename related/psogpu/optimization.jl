# import Pkg;
#import Pkg

#Pkg.add("CUDA")
#Pkg.add("SciMLBase")
#Pkg.add("PSOGPU")
#Pkg.add("StaticArrays")


#Pkg.add(["CUDA", "SciMLBase"])
#Pkg.add("PSOGPU")
using Pkg
using StaticArrays, CUDA
using SciMLBase 
#Pkg.add("PSOGPU")
using PSOGPU
using Optimization
using OptimizationBase

using BenchmarkTools

println("imported errything")
#lb = @SArray [-1.0f0, -1.0f0, -1.0f0]
#ub = @SArray [10.0f0, 10.0f0, 10.0f0, 10.0f0, 10.0f0, ]
lb = @SArray fill(-1.5f0, 10)
ub = @SArray fill(10.0f0, 10)

function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end

function rastrigin(x, p)
    sum(p[1] + x[i]^2 - p[1] * cos(p[2]*Float32(π) * x[i]) for i in 1:length(x))
end

x0 = @SArray zeros(Float32, 10)
p = @SArray Float32[10.0, 2.0]
 
prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n = 31622
#n = 1024 * 1024

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb=lb, ub=ub)
l0 = rastrigin(x0, p)
 
#sol = solve(prob, HybridPSO(;local_opt = PSOGPU.BFGS(), backend = CUDA.CUDABackend()), local_maxiters = 30)
#@show sol.objective

benchmark_result = @benchmark solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
time_taken = median(benchmark_result).time / 1e8

#sol = solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
sol = solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
#sol = solve(prob, ParallelSyncPSOKernel(1024*10243, backend = CUDA.CUDABackend()), maxiters = 500)
# Print timing and optimization results
println("Median execution time: ", time_taken, " seconds")
println("Optimal parameters found: ", sol.minimizer)
println("Minimum value of the objective function: ", sol.minimum)

#time_taken = @belapsed sol = solve(prob, ParallelSyncPSOKernel(1000, backend = CUDA.CUDABackend()), maxiters = 500)
#println("Execution time: ", time_taken, " seconds")

#@btime sol = solve(prob, ParallelSyncPSOKernel(1000, backend = CUDA.CUDABackend()), maxiters = 500)
#println("\nsol: ", sol);
