#include <vector>
#include <iostream>
#include "zeus.cuh"
#include "duals.cuh"
double
square(double x)
{
  return x * 2.5;
}

struct Rosen {
  template <class T, std::size_t N>
  __host__ __device__
  constexpr T operator()(std::array<T, N>const& x) const
  {
    T sum = T(0);
#pragma unroll
    for (int i = 0; i < N-1; ++i) {
      T t1 = T(1) - x[i];
      T t2 = x[i + 1] - x[i] * x[i];
      sum += t1 * t1 + T(100) * t2 * t2;
    }
    return sum;
  }
};

// templated Rastrigin on T (double or Dual) and N at compile time
struct Rast {
  template<class T, std::size_t N>
  __host__ __device__
  constexpr T operator()(const std::array<T,N>& x) const {
    T sum = T(10.0 * N);
  #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (std::is_same_v<T,double>) {
        sum += x[i]*x[i]
             - T(10.0) * T(std::cos(2.0 * M_PI * x[i]));
      } else {
        sum += x[i]*x[i]
             - T(10.0) * dual::cos(x[i] * T(2.0 * M_PI));
      }
    }
    return sum;
  }
};


int
main()
{
  std::vector ys{1.5, 2.5, 3.5};

  auto result = zeus::fmap(square, ys);
  for (auto val : result) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  int N = 131072;
  double host[N];
  for (int i = 0; i < N; i++) {
    host[i] = 333777.0;
  }
  util::set_stack_size();

  std::array<double,10> x10{};
  auto res10 = zeus::Zeus(Rosen{},x10, -5.12, 5.12, host, N, 10000, 5, 100, "rosenbrock", 1e-8, 42, 0);
  std::array<double, 3> x3{};
  auto res3 = zeus::Zeus(Rosen{},x3, -5.12, 5.12, host, N, 10000, 5, 100, "rosenbrock", 1e-8, 42, 0);
  std::cout << "global minimum for 10d rosenbrock: " << res10.fval << std::endl;
  std::cout << "global minimum for 3d rosenbrock: " << res3.fval << std::endl;

  std::array<double, 5> x5{};
  auto res5 = zeus::Zeus(Rast{},x5, -5.12, 5.12, host, N, 10000, 25, 100, "rastrigin", 1e-8, 42, 0);
  std::cout << "global minimum for 5d rastrigin: " << res5.fval << std::endl;
 
}
