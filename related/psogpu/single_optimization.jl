#!/usr/bin/env julia
using BenchmarkTools, PSOGPU, StaticArrays, KernelAbstractions, Optimization, CUDA, SciMLBase, LinearAlgebra
using Printf

@inline function mycos(x::Float32)
    # Using the built-in cosf for Float32 inputs
    return cos(x)
end

function rastrigin(x, _)
    A = 10.0f0
    # For 2D: N = 2, ensure all literals are Float32
    term1 = A * 2.0f0
    term2 = x[1]^2 - A * mycos(2f0 * π * x[1])
    term3 = x[2]^2 - A * mycos(2f0 * π * x[2])
    return term1 + term2 + term3
end

function ackley(x, _)
    a = 20.0f0
    b = 0.2f0
    c = 2f0 * π
    # For 2D: N = 2, work entirely in Float32
    s1 = x[1]^2 + x[2]^2
    s2 = mycos(c * x[1]) + mycos(c * x[2])
    term1 = -a * exp(-b * sqrt(s1 / 2f0))
    term2 = -exp(s2 / 2f0)
    return term1 + term2 + a + exp(1f0)
end


# Define the five functions (all in ℝ²)
#function ackley(x, _)
#    a = 20.0; b = 0.2; c = 2π; N = length(x)
#    term1 = -a * exp(-b * sqrt(sum(x.^2)/N))
#    term2 = -exp(sum(cos.(c .* x))/N)
#    return term1 + term2 + a + exp(1)
#end

function goldstein(x, _)
    x1, x2 = x[1], x[2]
    term1 = 1 + (x1 + x2 + 1)^2*(19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2)
    term2 = 30 + (2*x1 - 3*x2)^2*(18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2)
    return term1 * term2
end

function himmelblau(x, _)
    x1, x2 = x[1], x[2]
    return (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
end


#function rastrigin(x, _)
#    A = 10.0; N = length(x)
#    return A*N + sum(xi^2 - A*cos(2π*xi) for xi in x)
#end

function rosenbrock(x, p)
    return p[2]*(x[2] - x[1]^2)^2 + (p[1] - x[1])^2
end

# Dictionary of functions and their known optima in ℝ².
# (For himmelblau, we choose (3.0,2.0) as the reference.)
functions = Dict(
    "ackley"     => (f = ackley,     optimum = SVector{2,Float32}(0.0,0.0),     param = nothing),
    "goldstein"  => (f = goldstein,  optimum = SVector{2,Float32}(0.0,-1.0),    param = nothing),
    "himmelblau" => (f = himmelblau, optimum = SVector{2,Float32}(3.0,2.0),     param = nothing),
    "rastrigin"  => (f = rastrigin,  optimum = SVector{2,Float32}(0.0,0.0),     param = nothing),
    "rosenbrock" => (f = rosenbrock, optimum = SVector{2,Float32}(1.0,1.0),     param = SVector{2,Float32}(1.0,100.0))
)

# Problem dimension (set to 2 for all functions)
N = 2
max_iters = 100
# Initial guess
x0 = @SArray zeros(Float32, N)

# Single experiment run over all 5 functions with one particle.
function run_all(alg::String, n::Int)
    folder = "./data/$(alg)"
    isdir(folder) || mkpath(folder)
    filepath = joinpath(folder, "single_results.tsv")
    open(filepath, "a") do io
        for (name, data) in functions
            f = data.f
            optimum = data.optimum
            p = data.param
            # Create the OptimizationProblem
            if isnothing(p)
                optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, x0)
            else
                optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, x0, p)
            end

            sol = nothing
            benchmark_result = nothing

            if alg == "ParallelPSOKernel"
                fun = () -> solve(prob, ParallelPSOKernel(n, backend=CUDA.CUDABackend()), maxiters=max_iters)
                benchmark_result = @benchmark $fun()
                sol = fun()
            elseif alg == "HybridPSO"
                fun = () -> solve(prob, HybridPSO(; backend=CUDA.CUDABackend(), pso=ParallelPSOKernel(n, backend=CUDA.CUDABackend())), maxiters=max_iters)
                benchmark_result = @benchmark $fun()
                sol = fun()
            else
                fun = () -> solve(prob, ParallelSyncPSOKernel(n, backend=CUDA.CUDABackend()), maxiters=max_iters)
                benchmark_result = @benchmark $fun()
                sol = fun()
            end

            time_taken = median(benchmark_result).time / 1e9
            err = norm(sol.minimizer - optimum)
            @printf("Function: %s, Particles: %d\n", name, n)
            @printf("Median execution time: %.10f seconds\n", time_taken)
            @printf("Optimal parameters found: %s\n", join(sol.minimizer, ","))
            @printf("Minimum value: %.10f\n", sol.minimum)
            @printf("Error: %.17f\n\n", err)

            # Write: function, particles, time, error, minimum, found minimizer
            @printf(io, "%s\t%d\t%.10f\t%.17f\t%.10f\t%s\n", name, n, time_taken, err, sol.minimum, join(sol.minimizer, ","))
        end
    end
    println("Saved aggregated results to: $filepath")
end

# Main: command-line arguments or run single experiment with one particle.
if length(ARGS) >= 2
    alg = ARGS[1]  # "ParallelPSOKernel", "HybridPSO", or "ParallelSyncPSOKernel"
    n = parse(Int, ARGS[2])
    run_all(alg, n)
else
    alg = "ParallelPSOKernel"
    # Run a single optimization (1 particle) across all functions.
    run_all(alg, 1)
end

