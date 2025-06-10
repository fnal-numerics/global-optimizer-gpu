#pragma once
#include "dual_numbers.cuh"

namespace util {

// vector ops (exported with C linkage in your .cu)
extern "C" {
  __device__ __noinline__ void vector_add(const double*, const double*, double*, int);
  __device__ __noinline__ void vector_scale(const double*, double, double*, int);
}

// benchmark functors
template<int DIM> __device__ dual::DualNumber rosenbrock(const dual::DualNumber*);
template<int DIM> __host__ __device__ double        rosenbrock(const double*);

template<int DIM> __device__ dual::DualNumber rastrigin(const dual::DualNumber*);
template<int DIM> __host__ __device__ double        rastrigin(const double*);

template<int DIM> __device__ dual::DualNumber ackley(const dual::DualNumber*);
template<int DIM> __host__ __device__ double        ackley(const double*);

template<int DIM> struct Rosenbrock {
  __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
    return rosenbrock<DIM>(x);
  }
  __host__ __device__ static double evaluate(const double* x) {
    return rosenbrock<DIM>(x);
  }
};

template<int DIM> struct Rastrigin {
  __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
    return rastrigin<DIM>(x);
  }
  __host__ __device__ static double evaluate(const double* x) {
    return rastrigin<DIM>(x);
  }
};

template<int DIM> struct Ackley {
  __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
    return ackley<DIM>(x);
  }
  __host__ __device__ static double evaluate(const double* x) {
    return ackley<DIM>(x);
  }
};

// AD gradient
template<typename Function, int DIM>
__device__ void calculateGradientUsingAD(double* x, double* gradient);

} // namespace util

