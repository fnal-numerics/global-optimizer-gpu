#include "fun.h"
#include "duals.cuh"
#include "utils.cuh"
#include "pso.cuh"
#include "bfgs.cuh"
#include "zeus.cuh"

__device__ int d_stopFlag         = 0;
__device__ int d_convergedCount   = 0;
__device__ int d_threadsRemaining = 0;



template <int dim>
void
selectAndRunOptimization(double lower,
                         double upper,
                         double* hostResults,
                         int N,
                         int MAX_ITER,
                         int PSO_ITERS,
                         int requiredConverged,
                         double tolerance,
                         int seed,
                         const int run)
{
  int choice;
  std::cout << "\nSelect function to optimize:\n"
            << " 1. Rosenbrock\n"
            << " 2. Rastrigin\n"
            << " 3. Ackley\n";
  // Only show 2D-only options when dim == 2.
  if constexpr (dim == 2) {
    std::cout << " 4. GoldsteinPrice\n"
              << " 5. Eggholder\n"
              << " 6. Himmelblau\n";
  }
  std::cout
    << " 7. Custom (user-defined objective via expression or kernel file)\n"
    << "Choice: ";
  std::cin >> choice;
  std::cin.ignore();

  switch (choice) {
    case 1:
      std::cout << "\n\n\tRosenbrock Function\n" << std::endl;
      zeus::runOptimizationKernel<util::Rosenbrock<dim>, dim>(lower,
                                                        upper,
                                                        hostResults,
                                                        N,
                                                        MAX_ITER,
                                                        PSO_ITERS,
                                                        requiredConverged,
                                                        "rosenbrock",
                                                        tolerance,
                                                        seed,
                                                        run);
      break;
    case 2:
      std::cout << "\n\n\tRastrigin Function\n" << std::endl;
      zeus::runOptimizationKernel<util::Rastrigin<dim>, dim>(lower,
                                                       upper,
                                                       hostResults,
                                                       N,
                                                       MAX_ITER,
                                                       PSO_ITERS,
                                                       requiredConverged,
                                                       "rastrigin",
                                                       tolerance,
                                                       seed,
                                                       run);
      break;
    case 3:
      std::cout << "\n\n\tAckley Function\n" << std::endl;
      zeus::runOptimizationKernel<util::Ackley<dim>, dim>(lower,
                                                    upper,
                                                    hostResults,
                                                    N,
                                                    MAX_ITER,
                                                    PSO_ITERS,
                                                    requiredConverged,
                                                    "ackley",
                                                    tolerance,
                                                    seed,
                                                    run);
      break;
    case 4:
      if constexpr (dim != 2) {
        std::cerr
          << "Error: GoldsteinPrice is defined for 2 dimensions only.\n";
      } else {
        std::cout << "\n\n\tGoldsteinPrice Function\n" << std::endl;
        zeus::runOptimizationKernel<util::GoldsteinPrice<dim>, dim>(lower,
                                                              upper,
                                                              hostResults,
                                                              N,
                                                              MAX_ITER,
                                                              PSO_ITERS,
                                                              requiredConverged,
                                                              "goldstein",
                                                              tolerance,
                                                              seed,
                                                              run);
      }
      break;
    case 5:
      if constexpr (dim != 2) {
        std::cerr << "Error: Eggholder is defined for 2 dimensions only.\n";
      } else {
        std::cout << "\n\n\tEggholder Function\n" << std::endl;
        zeus::runOptimizationKernel<util::Eggholder<dim>, dim>(lower,
                                                         upper,
                                                         hostResults,
                                                         N,
                                                         MAX_ITER,
                                                         PSO_ITERS,
                                                         requiredConverged,
                                                         "eggholder",
                                                         tolerance,
                                                         seed,
                                                         run);
      }
      break;
    case 6:
      if constexpr (dim != 2) {
        std::cerr << "Error: Himmelblau is defined for 2 dimensions only.\n";
      } else {
        std::cout << "\n\n\tHimmelblau Function\n" << std::endl;
        zeus::runOptimizationKernel<util::Himmelblau<dim>, dim>(lower,
                                                          upper,
                                                          hostResults,
                                                          N,
                                                          MAX_ITER,
                                                          PSO_ITERS,
                                                          requiredConverged,
                                                          "himmelblau",
                                                          tolerance,
                                                          seed,
                                                          run);
      }
      break;
    case 7:
      std::cout << "\n\n\tCustom User-Defined Function\n" << std::endl;
      // for a more complex custom function, one option is to let the user
      // provide a path to a cuda file and compile it at runtime.
      // runOptimizationKernel<UserDefined<dim>, dim>(lower, upper, hostResults,
      // hostIndices,
      //                                             hostCoordinates, N,
      //                                             MAX_ITER);
      break;
    default:
      std::cerr << "Invalid selection!\n";
      exit(1);
  }
}

// #ifndef UNIT_TEST
// #ifndef NO_MAIN
#if !defined(UNIT_TEST) && !defined(TABLE_GEN)
int
main(int argc, char* argv[])
{
  printf("Production main() running\n");
  if (argc != 10) {
    std::cerr
      << "Usage: " << argv[0]
      << " <lower_bound> <upper_bound> <max_iter> <pso_iters> <converged> "
         "<number_of_optimizations> <tolerance> <seed> <run>\n";
    return 1;
  }
  double lower = std::atof(argv[1]);
  double upper = std::atof(argv[2]);
  int MAX_ITER = std::stoi(argv[3]);
  int PSO_ITERS = std::stoi(argv[4]);
  int requiredConverged = std::stoi(argv[5]);
  int N = std::stoi(argv[6]);
  double tolerance = std::stod(argv[7]);
  int seed = std::stoi(argv[8]);
  int run = std::stoi(argv[9]);

  std::cout << "Tolerance: " << std::setprecision(10) << tolerance
            << "\nseed: " << seed << "\n";

  // const size_t N =
  // 128*4;//1024*128*16;//pow(10,5.5);//128*1024*3;//*1024*128;
  const int dim = 10;
  double hostResults[N]; // = new double[N];
  std::cout << "number of optimizations = " << N << " max_iter = " << MAX_ITER
            << " dim = " << dim << std::endl;

  double f0 = 333777; // initial function value

  // logic to set the stact size limit to 65 kB per thread
  size_t currentStackSize = 0;
  cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
  printf("Current stack size: %zu bytes\n", currentStackSize);
  size_t newStackSize = 64 * 1024; // 65 kB
  cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
  if (err != cudaSuccess) {
    printf("cudaDeviceSetLimit error: %s\n", cudaGetErrorString(err));
    return 1;
  } else {
    printf("Successfully set stack size to %zu bytes\n", newStackSize);
  } // end stack size limit

  char cont = 'y';
  while (cont == 'y' || cont == 'Y') {
    for (int i = 0; i < N; i++) {
      hostResults[i] = f0;
    }
    selectAndRunOptimization<dim>(lower,
                                  upper,
                                  hostResults,
                                  N,
                                  MAX_ITER,
                                  PSO_ITERS,
                                  requiredConverged,
                                  tolerance,
                                  seed,
                                  run);
    std::cout << "\nDo you want to optimize another function? (y/n): ";
    std::cin >> cont;
    std::cin.ignore();
  }

  // for(int i=0; i<N; i++) {
  //     hostResults[i] = f0;
  // }
  // selectAndRunOptimization<dim>(lower, upper, hostResults, hostIndices,
  // hostCoordinates, N, MAX_ITER);
  return 0;
}
#endif
