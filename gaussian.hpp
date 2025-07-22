#pragma once

#include <array>
#include <cuda_runtime.h>

#include "matrix.cuh"

// template for square matrix
//   type of the thing
//   NxN
// allocate memory

// function to access the elements
// for i,j

// matrix class template
// class constructor gets told the size of the array

// copy the buffer
// class manage its own resources


  // copy constructor, gotta call cudamalloc
 // copy assignment operator, not a constructor --> 

 // move assignment
// destructor
// assignment operator=();
// copy and swap idiom

  // constructor to allocate matrxi on host + device
  // destructor to get rid of the covariance matrix 



template <std::size_t N>
class Gaussian {
  Matrix<double> C;

public:
  // host constructor copy the covariance matrix into device memory
  Gaussian(std::array<std::array<double,N>,N> const& C_host) : C(N,N)
  {
    // fill the host buffer
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j)
        C(i,j) = C_host[i][j];
  }

  // device & host call-operator
  template <class T>
  __host__ __device__
  T operator()(std::array<T,N> const& x) const {
    T q = T(0);
    #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      #pragma unroll
      for (std::size_t j = 0; j < N; ++j) {
        // if cuda_arch
            auto a = C.data()[i*N + j];
            q += x[i] * T(a * x[j]);
        // else
        //  use host matrix
      }
    }
    return T(0.5) * q;
  }
};

