#include <vector>
#include <iostream>

double
square(double x)
{
  return x * x;
}

int
main()
{
  std::vector ys{1.5, 2.5, 3.5};

  auto result = zeus::fmap(square, ys);
  for (auto val : result) {
    std::cout << val << " ";
  }
}