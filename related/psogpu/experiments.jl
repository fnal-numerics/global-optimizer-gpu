#!/usr/bin/env julia
using BenchmarkTools, PSOGPU, StaticArrays, KernelAbstractions, Optimization, CUDA, SciMLBase
using Printf

# Rosenbrock function:
#   f(x) = \sum_{i=1}^{N-1} \Big[p_2\Big(x_{i+1}-x_i^2\Big)^2 + (p_1-x_i)^2\Big]
function rosenbrock(x, p)
    sum(p[2]*(x[i+1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x)-1))
end

# Problem setup
N = 10
x0 = @SArray zeros(Float32, N)
p = @SArray Float32[1.0, 100.0]
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p)

function run_experiment(alg::String, n::Int)
    println("Algorithm: $alg, Particles: $n")
    sol = nothing
    benchmark_result = nothing

    if alg == "ParallelPSOKernel"
        f = () -> solve(prob, ParallelPSOKernel(n, backend=CUDA.CUDABackend()), maxiters=500)
        benchmark_result = @benchmark $f()
        sol = f()
    elseif alg == "HybridPSO"
        f = () -> solve(prob, HybridPSO(; backend=CUDA.CUDABackend(), pso=ParallelPSOKernel(n, backend=CUDA.CUDABackend())), maxiters=500)
        benchmark_result = @benchmark $f()
        sol = f()
    else
        f = () -> solve(prob, ParallelSyncPSOKernel(n, backend=CUDA.CUDABackend()), maxiters=500)
        benchmark_result = @benchmark $f()
        sol = f()
    end

    time_taken = median(benchmark_result).time / 1e9
    println("Median execution time: $time_taken seconds")
    println("Optimal parameters found: ", sol.minimizer)
    println("Minimum value: ", sol.minimum)
    
    # Append to the same algorithm.tsv file in a dedicated folder
    folder = "./data/$alg"
    isdir(folder) || mkpath(folder)
    filepath = joinpath(folder, "algorithm.tsv")
    open(filepath, "a") do io
        @printf(io, "%d\t%.6f\t%.6f\t%s\n", n, time_taken, sol.minimum, join(sol.minimizer, ","))
    end
    println("Saved results to: $filepath")
end

# Main: command-line arguments or batch experiment.
if length(ARGS) >= 2
    alg = ARGS[1]  # "ParallelPSOKernel", "HybridPSO", or "ParallelSyncPSOKernel"
    n = parse(Int, ARGS[2])
    run_experiment(alg, n)
else
    alg = "ParallelPSOKernel"
    for n in [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288]
        run_experiment(alg, n)
    end
end

