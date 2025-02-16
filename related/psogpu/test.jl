using Pkg
using CUDA
using StaticArrays
using SciMLBase
using Optimization
using PSOGPU

lb = @SArray [-1.0f0, -1.0f0, -1.0f0]
ub = @SArray [10.0f0, 10.0f0, 10.0f0]

function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end

x0 = @SArray zeros(Float32, 3)
p = @SArray Float32[1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)
sol = solve(prob,
            ParallelSyncPSOKernel(1000, backend = CUDA.CUDABackend()),
            maxiters = 500)
