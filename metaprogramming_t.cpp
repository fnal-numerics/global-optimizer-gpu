#include <vector>
#include <iostream>
#include "zeus.cuh" 

double
square(double x)
{
  return x * 2.5;
}

static constexpr int d = 10;

struct Rosen
{
    static constexpr int DIM = d;

    template<class T>
    __host__ __device__
    T operator()(const T* x) const
    {
        T sum = T(0);
        #pragma unroll
        for (int i = 0; i < d-1; ++i) {
            T t1 = T(1)   - x[i];
            T t2 = x[i+1] - x[i]*x[i];
            sum += t1*t1 + T(100)*t2*t2;
        }
        return sum;
    }
};

void set_stack_size() {
  // logic to set the stact size limit to 65 kB per thread
  size_t currentStackSize = 0;
  cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
  //printf("Current stack size: %zu bytes\n", currentStackSize);
  size_t newStackSize = 64 * 1024; // 65 kB
  cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
  if (err != cudaSuccess) {
    printf("cudaDeviceSetLimit error: %s\n", cudaGetErrorString(err));
    //return 1;
  }  
}

int
main()
{
  std::vector ys{1.5, 2.5, 3.5};

  auto result = zeus::fmap(square, ys);
  for (auto val : result) {
    std::cout << val << " ";
  }
  std::cout << std::endl;  
  
  int N = 1024;
  double host[N];
  for (int i = 0; i < N; i++) {
      host[i] = 333777.0;
  }
  util::set_stack_size();

  auto res = zeus::Zeus(Rosen{},-5.12, 5.12,host,1024,10000,5,100,"rosenbrock",1e-8,42,0);
  std::cout << "global minimum for rosenbrock: "<<res.fval <<std::endl;
  
}
