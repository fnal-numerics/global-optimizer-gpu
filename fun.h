#pragma once
#include "duals.cuh"

namespace util {

extern "C" {
  __device__ __noinline__ void vector_add(const double*, const double*, double*, int);
  __device__ __noinline__ void vector_scale(const double*, double, double*, int);
}

template<int DIM>
__device__ dual::DualNumber rosenbrock(const dual::DualNumber* x) {
  dual::DualNumber sum(0.0, 0.0);
  for (int i = 0; i < DIM - 1; ++i) {
    dual::DualNumber t1 = dual::DualNumber(1.0, 0.0) - x[i];
    dual::DualNumber t2 = x[i+1] - x[i]*x[i];
    sum = sum + t1*t1 + dual::DualNumber(100.0, 0.0)*t2*t2;
  }
  return sum;
}

template<int DIM>
__host__ __device__ double rosenbrock(const double* x) {
  double sum = 0.0;
  for (int i = 0; i < DIM - 1; ++i) {
    double t1 = 1.0 - x[i];
    double t2 = x[i+1] - x[i]*x[i];
    sum += t1*t1 + 100.0*t2*t2;
  }
  return sum;
}

template<int DIM>
__device__ dual::DualNumber rastrigin(const dual::DualNumber* x) {
  dual::DualNumber sum(10.0*DIM, 0.0);
  for (int i = 0; i < DIM; ++i) {
    sum = sum + ( x[i]*x[i]
                - dual::DualNumber(10.0,0.0)*dual::cos(x[i]*dual::DualNumber(2.0*M_PI,0.0)) );
  }
  return sum;
}

template<int DIM>
__host__ __device__ double rastrigin(const double* x) {
  double sum = 10.0*DIM;
  for (int i = 0; i < DIM; ++i)
    sum += x[i]*x[i] - 10.0*std::cos(2.0*M_PI*x[i]);
  return sum;
}

// Ackley Function (general d-dimensions)
//   f(x) = -20 exp\Bigl(-0.2\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}\Bigr)
//          - exp\Bigl(\frac{1}{d}\sum_{i=1}^{d}\cos(2\pi x_i)\Bigr)
//          + 20 + e
template<int DIM>
__device__
dual::DualNumber ackley(const dual::DualNumber* x) {
    dual::DualNumber sum_sq = 0.0;
    dual::DualNumber sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum_sq += dual::pow(x[i], 2);
        sum_cos += dual::cos(2.0 * M_PI * x[i]);
    }
    dual::DualNumber term1 = dual::DualNumber(-20.0) * dual::exp(-0.2 * dual::sqrt(sum_sq / DIM));
    dual::DualNumber term2 = dual::DualNumber(0.0) - dual::exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + dual::exp(1.0);
}

template<int DIM>
__host__ __device__
double ackley(const double* x) {
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum_sq += x[i] * x[i];
        sum_cos += cos(2.0 * M_PI * x[i]);
    }
    double term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / DIM));
    double term2 = -exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + exp(1.0);
}



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

template<typename Function, int DIM>
__device__ void calculateGradientUsingAD(double *x, double *gradient) {
    dual::DualNumber xDual[DIM];

    for (int i = 0; i < DIM; ++i) { // // iterate through each dimension (vairbale)
        xDual[i] = dual::DualNumber(x[i], 0.0);
    }

    // calculate the partial derivative of  each dimension
    for (int i = 0; i < DIM; ++i) {
        xDual[i].dual = 1.0; // derivative w.r.t. dimension i
        dual::DualNumber result = Function::evaluate(xDual); // evaluate the function using AD
        gradient[i] = result.dual; // store derivative
        //printf("\nxDual[%d]: %f, grad[%d]: %f ",i,xDual[i].real,i,gradient[i]);
        xDual[i].dual = 0.0;
    }
}


} // namespace util

