#pragma once

#include <array>
#include <cuda_runtime.h>

template <std::size_t N>
__constant__ double d_C[N * N];

template <std::size_t N>
struct Gaussian {

  // host constructor copy the covariance matrix into constant memory
  Gaussian(std::array<std::array<double,N>,N> const& C_host) {
    cudaMemcpyToSymbol(
      d_C<N>,         
      C_host.data(),           
      sizeof(double)*N*N          
    );
  }
  // device & host call-operator read from d_C
  template <class T>
  __host__ __device__
  T operator()(std::array<T,N> const& x) const {
    T q = T(0);
    #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      #pragma unroll
      for (std::size_t j = 0; j < N; ++j) {
        q += x[i] * T(d_C<N>[i*N + j] * x[j]);
      }
    }
    return T(0.5) * q;
  }
};

