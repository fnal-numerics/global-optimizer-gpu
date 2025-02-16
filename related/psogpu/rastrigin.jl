using Pkg

using PSOGPU, StaticArrays, KernelAbstractions, Optimization
using CUDA

using SciMLBase
using OptimizationBase
using BenchmarkTools

println("Imported everything")

lb = @SArray fill(-1.5f0, 10)
ub = @SArray fill(10.0f0, 10)


# generic function: dimensionality of function is determined
# by the length of the argument.
# In C++ I would have written a variadic template with a
# parameter pack that takes 1 or more arguments; I don’t
# know enough about Julia to know if there is an equivalent.
function rast(x::Vector{Float64}; A::Float64 = 10.0)
    N = length(x)  # number of dimensions is implicit
    return A * N + sum(xi^2 - A * cos(2 * π * xi) for xi in x)
end

function r1(x, p)
   sum(xi^2 - p[1] * cos(2 * π * xi) for xi in x)
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
p = @SArray Float32[1.0, 100.0]
#prob = OptimizationProblem(r1, x0, p; lb = lb, ub = ub)

n = 8

optf = OptimizationFunction(r1, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb = lb, ub = ub)
l0 = r1(x0, p)

bench = @benchmark solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)
time_taken = median(bench).time

println("Median execution time: ", time_taken, " ms")

sol = solve(prob, ParallelSyncPSOKernel(n, backend = CUDA.CUDABackend()), maxiters = 500)

println("Optimal parameters found: ", sol.minimizer)
println("Minimum value of the objective function: ", sol.minimum)
