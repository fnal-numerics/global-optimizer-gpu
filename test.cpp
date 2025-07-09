#include <vector>
#include "zeus.cuh"

double square(double x) {
  return x*x;
}

int main() {
  std::vector<double> ys{1,2,3};

  auto result = zeus::Zeus(square,-5.12, 5.12, ys);
  return 0;
}
