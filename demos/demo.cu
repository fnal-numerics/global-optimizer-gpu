#include <stdio.h>
#include "util.cuh"

using namespace std;
 
 __global__ void hello_world(int *x){
   printf("hello from gpu\n");
   for(int i=0; i<10; i++)
    printf("%i\t", x[i]);
   printf("\n");
}

int main(){
  
  cudaDeviceReset();
  int *dx, *dx2, *dx3;
  
  int x[10]  = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10};
  int x2[10] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  int x3[10] = {-11, -22, -3, -4, -5, -6, -7, -8, -9, -10};
  
  cudaMalloc((void **)&dx, sizeof(int)*10);
  cudaMemcpy(dx, x, sizeof(int)*10, cudaMemcpyHostToDevice);
  
  cudaMalloc((void **)&dx2, sizeof(int)*10);
  cudaMemcpy(dx2, x2, sizeof(int)*10, cudaMemcpyHostToDevice);
  
  cudaMalloc((void **)&dx3, sizeof(int)*10);
  cudaMemcpy(dx3, x3, sizeof(int)*10, cudaMemcpyHostToDevice);
  
  constexpr size_t numGPUArrays = 3;
  size_t gpuArraySize = 10;
  display<int, numGPUArrays>(gpuArraySize, dx, dx2, dx3);

  return 0;
}












