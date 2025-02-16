using Pkg
using CUDA
using StaticArrays
using SciMLBase
using PSOGPU
using Optimization
using OptimizationBase
using BenchmarkTools

println("Imported everything")

lb = @SArray fill(-1.5f0, 10)
ub = @SArray fill(10.0f0, 10)


function r1(x, p)
   p[1]*10.0 + sum(x[i]^2 - p[1] * cos(p[2] * Float32(π) * x[i]) for i in 1:length(x))
end

function rastrigin(x, p)
    p1 = p[1] # 10
    p2 = p[2] * Float32(π)
    s = 0.0
    
    #return p1*length(x) + sum(x[i]^2 - p1 * cos(p2 * x[i]) for i in 1:length(x))
    for xi in x
        s += p1 + xi^2 - p1 * cos(p2 * xi)
    end
    return s
end

x0 = @SArray zeros(Float32, 10)
p = @SArray Float32[10.0f0, 2.0f0]

prob = OptimizationProblem(rastrigin, x0, p; lb = lb, ub = ub)

n = 3162

optf = OptimizationFunction(rastrigin, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb = lb, ub = ub)
l0 = rastrigin(x0, p)

bench = @benchmark solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
time_taken = median(bench).time

println("Median execution time: ", time_taken, " ms")

# Use CPU backend for now
sol = solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)

println("Optimal parameters found: ", sol.minimizer)
println("Minimum value of the objective function: ", sol.minimum)

