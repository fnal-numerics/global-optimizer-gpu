#include <vector>
#include "zeus.cuh"

// To get the equivalent of the following function:
//
// double square(double x) {
//   return x*x;
// }
// We use the template below:

template <typename T>
auto
square_aux(T x)
{
  return x * x;
}

using square = square_aux<double>;

int
main()
{
  std::vector<double> ys{1, 2, 3};

  auto result = zeus::Zeus(square, -5.12, 5.12, ys);
  return 0;
}
