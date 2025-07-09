#include <vector>
#include <iostream>
#include "zeus.cuh" 

double
square(double x)
{
  return x * 2.5;
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
}
