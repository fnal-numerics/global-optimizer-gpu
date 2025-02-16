using Pkg

using SciMLBase

using PSOGPU, StaticArrays, KernelAbstractions, Optimization
using CUDA

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

#function rastrigin(x, p)
#    sum(p[1] + x[i]^2 - p[1] * cos(p[2]*Float32(π) * x[i]) for i in 1:length(x)-1))
#end
N = 10
x0 = @SArray zeros(Float32, 10)
p = @SArray Float32[1.0, 100.0]
x0 = @SArray zeros(Float32, N)
p = @SArray Float32[1.0, 100.0]
lb = @SArray fill(Float32(-1.0), N)
ub = @SArray fill(Float32(10.0), N)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb = lb, ub = ub)

#prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n = 512
println("number of particles: ", n)
#n = 1024 * 1024
 
#optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
#prob = OptimizationProblem(rosenbrock, x0, p; lb=lb, ub=ub)
l0 = rosenbrock(x0, p)

sol = solve(prob, PSOGPU.HybridPSO(; backend = CUDABackend() ,pso = PSOGPU.ParallelPSOKernel(n, backend = CUDA.CUDABackend())), maxiters = 500)
 
#sol = solve(prob, PSOGPU.HybridPSO(; backend = CUDABackend() ,pso = PSOGPU.ParallelPSOKernel(n;global_update = false,backend = CUDABackend()),local_opt = PSOGPU.LBFGS()), maxiters = 500,local_maxiters = 30)
@show sol.objective
@show sol.stats.time


#sol = solve(prob, HybridPSO( backend = CUDA.CUDABackend(), local_opt = PSOGPU.BFGS())) # local_maxiters = 30)
#@show sol.objective

benchmark_result = @benchmark solve(prob, ParallelPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
time_taken = median(benchmark_result).time / 1e9

#sol = solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
sol = solve(prob, ParallelPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
#sol = solve(prob, ParallelSyncPSOKernel(1024*10243, backend = CUDA.CUDABackend()), maxiters = 500)
# Print timing and optimization results
println("Median execution time: ", time_taken, " seconds")
println("Optimal parameters found: ", sol.minimizer)
println("Minimum value of the objective function: ", sol.minimum)
