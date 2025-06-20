#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <cstring>

#include <filesystem>


#include "fun.cuh"
#include "duals.cuh"

// https://xorshift.di.unimi.it/splitmix64.c
// Very fast 64-bit mixer — returns a new 64-bit value each time.
__device__ inline uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL); // 1 add
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; // 1 shift, 1 xor, 1 64x64 multiplier
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; // 1 shift, 1, xor, 1 64x64 multiplier
    //printf("split");
    return z ^ (z >> 31); // 1 shift, 1 xor
}

// return a random double in [minVal, maxVal)
__device__ inline double random_double(uint64_t &state,
                                       double minVal,
                                       double maxVal) {
    // get 64‐bit random int
    uint64_t z = splitmix64(state);
    // map high 53 bits into [0,1)
    double u = (z >> 11) * (1.0 / 9007199254740992.0); // discard lower 11 bits, leaving mantissa width of IEEE double, then normalize integer into [0,1)
    // scale into [minVal, maxVal)
    return minVal + u * (maxVal - minVal);
}

namespace util {

__device__ double dot_product_device(const double* a, const double* b, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__device__ void outer_product_device(const double* v1, const double* v2, double* result, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
	    int idx = i * size + j;
	    if (idx < size*size){
		result[idx] = v1[i] * v2[j];    
            } else {
   	        printf("outer product out of bounds..\ndim=%d i*size+j=%d\n",size, i * size + j);
            }
	}
    }
}

template<int DIM>
__device__ double calculate_gradient_norm(const double* g) {
    double grad_norm = 0.0;
    for (int i = 0; i < DIM; ++i) {
        grad_norm += g[i] * g[i];
    }
    return sqrt(grad_norm);
}

template<int DIM>
__device__ void compute_search_direction(double* p,const double*  H,const double* g) {
    for (int i = 0; i < DIM; i++) {
        double sum=0.0;
        for (int j = 0; j < DIM; j++) {
           sum += H[i * DIM + j] * g[j]; // i * dim + j since H is flattened arr[]
        }    
    p[i] = -sum;
    }
}

// wrap kernel definitions extern "C" block so that their symbols are exported with C linkage
extern "C" {
__device__ __noinline__ void vector_add(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

__device__ __noinline__ void vector_scale(const double* a, double scalar, double* result, int dim) {
    for (int i = 0; i < dim; ++i) {
        result[i] = a[i] * scalar;
    }
}
}// end extern C

__device__ void initialize_identity_matrix(double* H, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            H[i * dim + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

__device__ bool valid(double x) {
    if (isinf(x)) {
        return false;
    } else if (isnan(x)) {
	return false;
    } else {
        return true;
    }
}


__device__ double pow2(double x) {
    return x * x;
}


__device__ void initialize_identity_matrix_device(double* H, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

template<int DIM>
__device__ void matrix_multiply_device(const double* A, const double* B, double* C) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < DIM; ++k) {
                sum += A[i * DIM + k] * B[k * DIM + j];
            }
            C[i * DIM + j] = sum;
        }
    }
}


// BFGS update with compile-time dimension
template<int DIM>
__device__ void bfgs_update(double* H, const double* s, const double* y, double sTy) {
    if (::fabs(sTy) < 1e-14) return;
    double rho = 1.0 / sTy;
    
    // Compute H_new element-wise without allocating large temporary matrices.
    // H_new = (I - rho * s * y^T) * H * (I - rho * y * s^T) + rho * s * s^T
    double H_new[DIM * DIM];  // Temporary array (DIM^2 elements)
    
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            double sum = 0.0;
            for (int k = 0; k < DIM; k++) {
                // Compute element (i,k) of (I - rho * s * y^T)
                double A_ik = ((i == k) ? 1.0 : 0.0) - rho * s[i] * y[k];
                double inner = 0.0;
                for (int m = 0; m < DIM; m++) {
                    // Compute element (m,j) of (I - rho * y * s^T)
                    double B_mj = ((m == j) ? 1.0 : 0.0) - rho * y[m] * s[j];
                    inner += H[k * DIM + m] * B_mj;
                }
                sum += A_ik * inner;
            }
            // Add the rho * s * s^T term
            H_new[i * DIM + j] = sum + rho * s[i] * s[j];
        }
    }
    
    // Copy H_new back into H
    for (int i = 0; i < DIM * DIM; i++) {
        H[i] = H_new[i];
    }
}


// function to calculate scalar directional direvative d = g * p
__device__ double directional_derivative(const double *grad, const double *p, int dim) {
    double d = 0.0;
    for (int i = 0; i < dim; ++i) {
        d += grad[i] * p[i];
    }
    return d;
}


// Goldstein-Price Function
//   f(x,y) = [1+(x+y+1)^2 (19-14x+3x^2-14y+6xy+3y^2)]
//            [30+(2x-3y)^2 (18-32x+12x^2+48y-36xy+27y^2)]
template<int DIM>
__device__
dual::DualNumber goldstein_price(const dual::DualNumber* x) {
    static_assert(DIM == 2, "Goldstein-Price is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0];
    dual::DualNumber x2 = x[1];
    dual::DualNumber term1 = dual::DualNumber(1.0) + dual::pow(x1 + x2 + 1.0, 2) *
        (19.0 - 14.0 * x1 + 3.0 * dual::pow(x1, 2) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * dual::pow(x2, 2));
    dual::DualNumber term2 = dual::DualNumber(30.0) + dual::pow(2.0 * x1 - 3.0 * x2, 2) *
        (18.0 - 32.0 * x1 + 12.0 * dual::pow(x1, 2) + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * dual::pow(x2, 2));
    return term1 * term2;
}

template<int DIM>
__host__ __device__
double goldstein_price(const double* x) {
    static_assert(DIM == 2, "Goldstein-Price is defined for 2 dimensions only.");
    double x1 = x[0];
    double x2 = x[1];
    double term1 = 1.0 + pow(x1 + x2 + 1.0, 2) *
        (19.0 - 14.0 * x1 + 3.0 * pow(x1, 2) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * pow(x2, 2));
    double term2 = 30.0 + pow(2.0 * x1 - 3.0 * x2, 2) *
        (18.0 - 32.0 * x1 + 12.0 * pow(x1, 2) + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * pow(x2, 2));
    return term1 * term2;
}

template<int DIM>
struct GoldsteinPrice {
    __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return goldstein_price<DIM>(x);
    }
    __host__ __device__ static double evaluate(const double* x) {
        return goldstein_price<DIM>(x);
    }
};

// Eggholder Function
//   f(x,y) = -(y+47) sin\Bigl(\sqrt{\Bigl|x/2+y+47\Bigr|}\Bigr)
//            - x sin\Bigl(\sqrt{\Bigl|x-(y+47)\Bigr|}\Bigr)

template<int DIM>
__device__ dual::DualNumber eggholder(const dual::DualNumber* x) {
    static_assert(DIM == 2, "Eggholder is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0], x2 = x[1];
    // Use (0 - value) in place of unary minus
    dual::DualNumber term1 = (dual::DualNumber(0.0) - (x2 + dual::DualNumber(47.0))) *
        dual::sin(dual::sqrt(dual_abs(x1 / dual::DualNumber(2.0) + x2 + dual::DualNumber(47.0))));
    dual::DualNumber term2 = (dual::DualNumber(0.0) - x1) *
        dual::sin(dual::sqrt(dual_abs(x1 - (x2 + dual::DualNumber(47.0)))));
    return term1 + term2;
}

template<int DIM>
__device__ double eggholder(const double* x) {
    static_assert(DIM == 2, "Eggholder is defined for 2 dimensions only.");
    double x1 = x[0];
    double x2 = x[1];
    double term1 = -(x2 + 47.0) *
        sin(sqrt(fabs(x1 / 2.0 + x2 + 47.0)));
    double term2 = -x1 *
        sin(sqrt(fabs(x1 - (x2 + 47.0))));
    return term1 + term2;
}

template<int DIM>
struct Eggholder {
    __host__ __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return eggholder<DIM>(x);
    }
    __host__ __device__ static double evaluate(const double* x) {
        return eggholder<DIM>(x);
    }
};


// Himmelblau's Function (2D only)
template<int DIM>
__device__
dual::DualNumber himmelblau(const dual::DualNumber* x) {
    static_assert(DIM == 2, "Himmelblau's function is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0], x2 = x[1];
    dual::DualNumber term1 = dual::pow(x1 * x1 + x2 - dual::DualNumber(11.0), 2);
    dual::DualNumber term2 = dual::pow(x1 + x2 * x2 - dual::DualNumber(7.0), 2);
    return term1 + term2;
}

template<int DIM>
__host__ __device__
double himmelblau(const double* x) {
    static_assert(DIM == 2, "Himmelblau's function is defined for 2 dimensions only.");
    double x1 = x[0], x2 = x[1];
    double term1 = pow(x1 * x1 + x2 - 11.0, 2);
    double term2 = pow(x1 + x2 * x2 - 7.0, 2);
    return term1 + term2;
}

template<int DIM>
struct Himmelblau {
    __host__ __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return himmelblau<DIM>(x);
    }
    __host__ __device__ static double evaluate(const double* x) {
        return himmelblau<DIM>(x);
    }
};

__device__ double generate_random_double(curandState* state, double lower,double upper)
{ 
    return lower + (upper + (-lower)) * curand_uniform_double(state);
}

__global__ void setup_curand_states(curandState* states, uint64_t seed, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &states[idx]);
}

template<typename Function, int DIM>
__device__ double line_search(double f0, const double* x, const double* p, const double* g){
    const double c1=0.3;
    double alpha=1.0;
    double ddir = dot_product_device(g,p,DIM);
    double xTemp[DIM];
    for(int i=0;i<20;i++){
        for(int j=0;j<DIM;j++){
            xTemp[j] = x[j] + alpha*p[j];
        }
        double f1 = Function::evaluate(xTemp);
        if(f1 <= f0 + c1*alpha*ddir) break;
        alpha *= 0.5;
    }
    return alpha;
}


} // util namespace end




__device__ __forceinline__
double atomicMinDouble(double* addr, double val) {
    // reinterpret the address as 64‑bit unsigned
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old_bits = *ptr, assumed_bits;

    do {
        assumed_bits = old_bits;
        double old_val = __longlong_as_double(assumed_bits);
        // if the current value is already <= our candidate, nothing to do
        if (old_val <= val) break;
        // else try to swap in the new min value’s bit‐pattern
        unsigned long long new_bits = __double_as_longlong(val);
        old_bits = atomicCAS(ptr, assumed_bits, new_bits);
    } while (assumed_bits != old_bits);

    // return the previous minimum
    return __longlong_as_double(old_bits);
}

// kernel #1: initialize X, V, pBest; atomically seed gBestVal/gBestX
template<typename Function, int DIM>
__global__ void psoInitKernel(
    Function           func,
    double             lower,
    double             upper,
    double*            X,
    double*            V,
    double*            pBestX,
    double*            pBestVal,
    double*            gBestX,
    double*            gBestVal,
    int                N,
    uint64_t	       seed,
    curandState*       states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    curandState localState = states[i];
    const double vel_range = (upper - lower) * 0.1;
    // const unsigned int seed = 1234u;
    //uint64_t counter = seed ^ (uint64_t)i;
    //uint64_t state = seed * 0x9e3779b97f4a7c15ULL + (uint64_t)i;
    //if (i==0) {
    //  printf(">> initKernel sees seed = %llu\n", (unsigned long long)seed);
    //}
    // init position & velocity
    for (int d = 0; d < DIM; ++d) {
        double rx = util::generate_random_double(&localState,lower, upper);
        double rv = util::generate_random_double(&localState,-vel_range, vel_range);
        //double rx = random_double(state, lower, upper); //  util::generate_random_double(seedX, lower, upper);
	//double rv = random_double(state, -vel_range, +vel_range); // util::generate_random_double(seedV, -vel_range, vel_range);
        
	X[i*DIM + d]      = rx;
        V[i*DIM + d]      = rv;
        pBestX[i*DIM + d] = rx;
    }

    // eval personal best
    double fval = Function::evaluate(&X[i*DIM]);
    pBestVal[i] = fval;
    
    // atomic update of global best
    double oldGB = atomicMinDouble(gBestVal, fval);
    if (fval < oldGB) {
        // we’re the new champion: copy pBestX into gBestX
        for (int d = 0; d < DIM; ++d)
            gBestX[d] = pBestX[i*DIM + d];
    }
    states[i] = localState; // next time we draw, we continue where we left off
}

// kernel #2: one PSO iteration (velocity+position update, personal & global best)
template<typename Function, int DIM>
__global__ void psoIterKernel(
    Function           func,
    double             lower,
    double             upper,
    double             w, // weight inertia
    double             c1, // cognitive coefficient
    double             c2, // social coefficient
    double*            X,
    double*            V,
    double*            pBestX,
    double*            pBestVal,
    double*            gBestX,
    double*            gBestVal,
    double*            traj,        // pass nullptr if not saving
    bool               saveTraj,
    int                N,
    int                iter,
    uint64_t	       seed,
    curandState*       states)//,
    //Result<DIM>&        best) // store the best particle's results at each iteration
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    curandState localState = states[i];
    /*uint64_t state = seed;
    state = state * 6364136223846793005ULL + iter;   // mix in iteration
    state = state * 6364136223846793005ULL + (uint64_t)i;  // mix in thread idx
    */

    // update velocity & position
    for (int d = 0; d < DIM; ++d) {
        //uint64_t z1 = splitmix64(state);
        //uint64_t z2 = splitmix64(state);
	//double r1 = random_double(state,0.0,1.0 ); //util::generate_random_double(seed1, 0.0, 1.0);
        //double r2 = random_double(state, 0.0, 1.0); //util::generate_random_double(seed2, 0.0, 1.0);
    	double r1 = util::generate_random_double(&localState, 0.0, 1.0);
        double r2 = util::generate_random_double(&localState, 0.0, 1.0);

	double xi = X[i*DIM + d];
        double vi = V[i*DIM + d];
        double pb = pBestX[i*DIM + d];
        double gb = gBestX[d];

        double nv = w*vi
                  + c1*r1*(pb - xi) // “cognitive” pull toward personal best
                  + c2*r2*(gb - xi); // “social” pull toward global best
        double nx = xi + nv;

        V[i*DIM + d] = nv;
        X[i*DIM + d] = nx;

        if (saveTraj) {
            // traj is laid out [iter][i][d]
            size_t idx = size_t(iter)*N*DIM + i*DIM + d;
            traj[idx] = nx;
        }
    }

    // evaluate at new position
    double fval = Function::evaluate(&X[i*DIM]);

    // personal best? no atomic needed, it's a private best position
    if (fval < pBestVal[i]) {
        pBestVal[i] = fval;
        for (int d = 0; d < DIM; ++d)
            pBestX[i*DIM + d] = X[i*DIM + d];
    }

    // global best?
    double oldGB = atomicMinDouble(gBestVal, fval);
    if (fval < oldGB) {
        for (int d = 0; d < DIM; ++d)
            gBestX[d] = X[i*DIM + d];
    }
    // best.coordinates = gBestX;
    // best.fval = gBestVal;
    /*printf("it %d gBestVal = %.6f  at gBestX = [",i,fval);
    for (int d = 0; d < DIM; ++d)
        printf(" %8.4f", gBestX[d]);
    printf(" ]\n");*/
    states[i] = localState; // next time we draw, we continue where we left off
}

template<int DIM>
struct Result {
    int idx;
    int status; // 1 if converged, else if stopped_bc_someone_flipped_the_flag: 2, else 0
    double fval; // function value
    double gradientNorm;
    double coordinates[DIM];
    int iter;
};

__device__ int d_stopFlag;  // 0 = keep going; 1 = stop immediately
__device__ int d_convergedCount; // how many threads have converged?
__device__ int d_threadsRemaining;

template<typename Function, int DIM, unsigned int blockSize>
__global__ void optimizeKernel(const double lower,const double upper,
		const double* __restrict__ pso_array, // pso initialized positions
		double* deviceResults, double* deviceTrajectory, int N,const int MAX_ITER,const int requiredConverged,const double tolerance, Result<DIM>* result, curandState* states, bool save_trajectories = false) {
    extern __device__ int d_stopFlag;
    extern __device__ int d_threadsRemaining;
    extern __device__ int d_convergedCount;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState localState = states[idx];

    //int early_stopping = 0;
    double H[DIM * DIM];
    double g[DIM], x[DIM], x_new[DIM], p[DIM], g_new[DIM], delta_x[DIM], delta_g[DIM];//, new_direction[DIM];
    //double tolerance = 1e-5;
    // Line Search params

    Result<DIM> r;
    r.status       = -1;     // assume “not converged” by default
    r.fval         = 333777.0;
    r.gradientNorm = 69.0;
    for (int d = 0; d < DIM; ++d) {
        r.coordinates[d] = 0.0;
    }
    r.iter = -1;
    r.idx = idx;
    util::initialize_identity_matrix(H, DIM);
    
    int num_steps = 0;

    // initialize x either from PSO array or fallback by RNG
    if (pso_array) {
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            x[d] = pso_array[idx*DIM + d];
            //if(idx == 0) printf("x[%d]=%0.7f\n", d, x[d]);
        }
    } else {
        //unsigned int seed = 456;
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            //x[d] = util::statelessUniform(idx,d,1,lower, upper, seed);
	    x[d] = util::generate_random_double(&localState, lower, upper);
        }
        states[idx] = localState;
    }	
    
    double f0 = Function::evaluate(x);//rosenbrock_device(x, DIM);
    deviceResults[idx] = f0;
    double bestVal = f0;
    //if (idx == 0) printf("\n\nf0 = %f", f0);
    int iter;
    util::calculateGradientUsingAD<Function, DIM>(x, g);
    for (iter = 0; iter < MAX_ITER; ++iter) {
        printf("inside BeeG File System");
        // check if somebody already asked to stop
	if (atomicAdd(&d_stopFlag, 0) != 0) { // atomicAdd here just to get a strong read-barrier 
            // CUDA will fetch a coherent copy of the integer from global memory. 
	    // as soon as one thread writes 1 into d_stopFlag via atomicExch, 
	    // the next time any thread does atomicAdd(&d_stopFlag, 0) it’ll see 1 and break.	   
            //printf("thread %d get outta dodge cuz we converged...", idx);
            r.status = 2;
            r.iter = iter;
	    r.fval = Function::evaluate(x);
            for(int d=0;d<DIM;d++){r.coordinates[d] = x[d];}
            r.gradientNorm = util::calculate_gradient_norm<DIM>(g); 
            break;
            
        }
	num_steps++;

        util::compute_search_direction<DIM>(p, H, g); //p = -H * g        

	// use the alpha obtained from the line search
	double alpha = util::line_search<Function,DIM>(bestVal, x, p, g);
	if(alpha == 0.0) {
            printf("Alpha is zero, no movement in iteration=%d\n", iter);
            alpha = 1e-3; 
        }

	// update current point x by taking a step size of alpha in the direction p
	for (int i = 0; i < DIM; ++i) {
            x_new[i] = x[i] + alpha * p[i];
	    delta_x[i] = x_new[i] - x[i];
	}

        double fnew = Function::evaluate(x_new);	
        // get the new gradient g_new at x_new
	util::calculateGradientUsingAD<Function, DIM>(x_new, g_new);
	
	// calculate new delta_x and delta_g
        for (int i = 0; i < DIM; ++i) {
	    delta_g[i] = g_new[i] - g[i]; // difference in gradient at the new point vs old point
        }

	// calculate the the dot product between the change in x and change in gradient using new point
        double delta_dot = util::dot_product_device(delta_x, delta_g, DIM);

	// bfgs update on H
        util::bfgs_update<DIM>(H, delta_x, delta_g, delta_dot);
        // only update x and g for next iteration if the new minima is smaller than previous
	//double min = Function::evaluate(x_new);//rosenbrock_device(x_new, DIM);
        if (fnew < bestVal) {
	   bestVal = fnew;
	   for(int i=0; i<DIM; ++i) {
	      x[i] = x_new[i];
	      g[i] = g_new[i];
	   } 
	}
        // refactor? yes 
        double grad_norm = util::calculate_gradient_norm<DIM>(g);
        if (grad_norm < tolerance) {
            // atomically increment the converged counter
            int oldCount = atomicAdd(&d_convergedCount, 1);
            int newCount = oldCount + 1;
            double fcurr = Function::evaluate(x);
            //printf("\nconverged for %d at iter=%d); f = %.6f;",idx, iter,fcurr);
            //for (int d = 0; d < DIM; ++d) { printf(" % .6f", x[d]);}
            //printf(" ]\n");
            r.status       = 1;
            r.gradientNorm = grad_norm;
            r.fval         = Function::evaluate(x);
            r.iter         = iter;
            for (int d = 0; d < DIM; ++d) {
                r.coordinates[d] = x[d];
            }	    
	    // if we just hit the threshold set by the user, the VERY FIRST thread to do so
            // sets d_stopFlag=1 so everyone else exits on their next check
            if (newCount == requiredConverged) {
                // flip the global stop flag
                atomicExch(&d_stopFlag, 1);
                __threadfence();
                printf("\nThread %d is the %d%s converged thread (iter=%d); fn = %.6f.\n",idx, newCount,(newCount==1? "st" : newCount==2? "nd" : newCount==3? "rd" : "th"),iter,fcurr);
            }
            // in _any_ case, whether we were the last to converge or not, 
            // we are individually done so break
            break;
        }

	//  deviceTrajectory layout: idx * (MAX_ITER * DIM) + iter * DIM + i
	if (save_trajectories) {
	   for (int i = 0; i < DIM; i++) {
               deviceTrajectory[idx * (MAX_ITER * DIM) + iter * DIM + i] = x[i];
           }
	} 

	//for(int i=0; i<DIM; ++i) {x[i] = x_new[i];}
    }// end bfgs loop
    // if we broek out because we hit the max numberof iterations, then its a surrender
    if(MAX_ITER == iter) {
        r.status = 0; // surrender
        r.iter = iter;
        r.gradientNorm = util::calculate_gradient_norm<DIM>(g);
        r.fval = Function::evaluate(x);
        for (int d = 0; d < DIM; ++d) { r.coordinates[d] = x[d];}
    }
    deviceResults[idx] = Function::evaluate(x);
    result[idx] = r;
}// end optimizerKernel

bool askUser2saveTrajectories() {
    std::cout << "Save optimization trajectories? (y/n): ";
    char ans;
    std::cin >> ans;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return (ans == 'y' || ans == 'Y');
}

void createOutputDirs(const std::string &path) {
    std::filesystem::create_directories(path);
}

cudaError_t writeTrajectoryData(
    double* hostTrajectory,
    int N, int MAX_ITER, int DIM,
    const std::string &fun_name,
    const std::string &basePath
) {
    // construct the directory path and create it.
    std::string dirPath = basePath + "/" + fun_name + "/"
       + std::to_string(DIM) + "d/" + std::to_string(MAX_ITER * N) + "/trajectories";
    std::filesystem::create_directories(dirPath);
    //createOutputDirs(dirPath);

    // the final filename.
    std::string filename = dirPath + "/"
                         + std::to_string(MAX_ITER) + "it_"
                         + std::to_string(N) + ".tsv";

    std::ofstream stepOut(filename);
    stepOut << "OptIndex\tStep";
    for (int d = 0; d < DIM; d++)
        stepOut << "\tX_" << d;
    stepOut << "\n";
    stepOut << std::scientific << std::setprecision(17);
    for (int i = 0; i < N; i++) {
        for (int it = 0; it < MAX_ITER; it++) {
            stepOut << i << "\t" << it;
            for (int d = 0; d < DIM; d++) {
                stepOut << "\t" << hostTrajectory[i * (MAX_ITER * DIM) + it * DIM + d];
            }
	    stepOut << "\n";
        }
    }
    stepOut.close();
    return cudaSuccess;
}


// make it write to std::cout + dump to file
template<int DIM>
void dump_data_2_file(const Result<DIM>* h_results,const std::string fun_name,const int N, const int PSO_ITER) {
    std::string filename = "./data/" + fun_name +"_" + std::to_string(PSO_ITER)+"psoit_" + std::to_string(DIM) + "d_particledata.tsv";

    bool file_exists = std::filesystem::exists(filename);
    bool file_empty = file_exists ? (std::filesystem::file_size(filename) == 0) : true;
    std::ofstream outfile(filename, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // if file is new or empty, let us write the header
    if (file_empty) {
        outfile << "fun\tidx\tstatus\titer\tfval\tnorm";
        for (int i = 0; i < DIM; i++)
            outfile << "\tcoord_" << i;
        outfile << std::endl;
    }// end if file is empty

    std::string tab = "\t";
    int countConverged = 0, surrender = 0, stopped = 0;
    for (int i = 0; i < N; ++i) {
        outfile << fun_name << tab << i << tab << std::scientific; 
        if (h_results[i].status == 1) {
            countConverged++;
            outfile << 1 << tab;
        } else if(h_results[i].status == 2) { // particle was stopped early
            stopped++;
            outfile << 2 << tab;
            //printf("Thread %d was stopped early (iter=%d)\n", i, h_results[i].iter);
        } else {
            surrender++;
            outfile << 0 << tab;
        }
        outfile << h_results[i].iter << tab << h_results[i].fval << tab << h_results[i].gradientNorm;
        for(int d = 0; d < DIM; ++d) {
            outfile << "\t"<< h_results[i].coordinates[d];
        }
        outfile << std::endl;
    }
    //std::cout << "\ndumped data 2 "<< filename << "\n"<<countConverged <<" converged, "<<stopped << " stopped early, "<<surrender<<" surrendered\n"; 
    //printf("\ndumped data 2 %s\n%d converged, %d stopped early, %d surrendered\n",filename.c_str(),countConverged, stopped, surrender);
}


void append_results_2_tsv(const int dim,const int N, const std::string fun_name,float ms_init, float ms_pso,float ms_opt,float ms_rand, const int max_iter, const int pso_iter,const double error,const double globalMin, double* hostCoordinates, const int idx, const int status, const double norm) {
        std::string filename = "zeus_" + std::to_string(dim) + "d_results.tsv";
        std::ofstream outfile(filename, std::ios::app);
        
        bool file_exists = std::filesystem::exists(filename);
        bool file_empty = file_exists ? (std::filesystem::file_size(filename) == 0) : true;
        //std::ofstream outfile(filename, std::ios::app);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // if file is new or empty, let us write the header
        if (file_empty) {
            outfile << "fun\tN\tidx\tstatus\tbfgs_iter\tpso_iter\ttime\terror\tfval\tnorm\t";
            for (int i = 0; i < dim; i++)
                outfile << "\tcoord_" << i;
            outfile << std::endl;
        }// end if file is empty
        
        double time_seconds = std::numeric_limits<double>::infinity();
        if (pso_iter > 0) {
            time_seconds = (ms_init+ms_pso+ms_opt+ms_rand);
            //printf("total time = pso + bfgs = total time = %0.4f ms\n", time_seconds);
        } else {
            time_seconds = (ms_opt+ms_rand);
            //printf("bfgs time = total time = %.4f ms\n", time_seconds);
        }
        outfile << fun_name << "\t" << N << "\t"<<idx<<"\t"<<status <<"\t" << max_iter << "\t" << pso_iter << "\t"
            << time_seconds << "\t"
            << std::scientific << error << "\t" << globalMin << "\t" << norm <<"\t" ;
        for (int i = 0; i < dim; i++) {
            outfile << hostCoordinates[i];
            if (i < dim - 1)
                outfile << "\t";
        }
        outfile << "\n";
        outfile.close();
        //printf("results are saved to %s", filename.c_str());
}// end append_results_2_tsv

template<typename Function, int DIM>
double* launch_pso(const int PSO_ITER,const int N,const double lower,const double upper, float& ms_init, float& ms_pso,const int seed, curandState* states) { //, Result<DIM>& best) {
        // allocate PSO buffers on device
        double *dX, *dV, *dPBestVal, *dGBestX, *dGBestVal, *dPBestX;
        cudaMalloc(&dX,        N*DIM*sizeof(double));
        cudaMalloc(&dV,        N*DIM*sizeof(double));
        cudaMalloc(&dPBestX,   N*DIM*sizeof(double));
        cudaMalloc(&dPBestVal, N   *sizeof(double));
        cudaMalloc(&dGBestX,   DIM *sizeof(double));
        cudaMalloc(&dGBestVal, sizeof(double));
        int zero = 0;
        cudaMemcpyToSymbol(d_stopFlag, &zero, sizeof(int));
        cudaMemcpyToSymbol(d_threadsRemaining, &N, sizeof(int));
        cudaMemcpyToSymbol(d_convergedCount,   &zero, sizeof(int));
        // set seed to infinity
        {
            double inf = std::numeric_limits<double>::infinity();
            cudaMemcpy(dGBestVal, &inf, sizeof(inf), cudaMemcpyHostToDevice);
        }

        dim3 psoBlock(256);
        dim3 psoGrid((N + psoBlock.x - 1) / psoBlock.x);

        // host-side buffers for printing
        double hostGBestVal;
        std::vector<double> hostGBestX(DIM);

        // PSO‐init Kernel
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        psoInitKernel<Function,DIM><<<psoGrid,psoBlock>>>(
             Function(), lower, upper,
             dX, dV,
             dPBestX, dPBestVal,
             dGBestX, dGBestVal,
             N,seed, states);
        cudaDeviceSynchronize();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        cudaEventElapsedTime(&ms_init, t0, t1);
        //printf("PSO‑Init Kernel execution time = %.4f ms\n", ms_init);

        // copy back and print initial global best
        cudaMemcpy(&hostGBestVal, dGBestVal, sizeof(double),        cudaMemcpyDeviceToHost);
        cudaMemcpy(hostGBestX.data(),  dGBestX,   DIM*sizeof(double), cudaMemcpyDeviceToHost);

        //printf("Initial PSO gBestVal = %.6e at gBestX = [", hostGBestVal);
        //for(int d=0; d<DIM; ++d) printf(" %.4f", hostGBestX[d]);
        //printf(" ]\n\n");

        // PSO iterations
        const double w  = 0.7298, c1 = 1.4962, c2 = 1.4962;
        //const double w  = 0.5, c1 = 1.2, c2 = 1.5;
        for(int iter=1; iter<PSO_ITER+1; ++iter) {
            cudaEventRecord(t0);
            psoIterKernel<Function,DIM><<<psoGrid,psoBlock>>>(
                Function(),
                lower, upper,
                w, c1, c2,
                dX, dV,
                dPBestX, dPBestVal,
                dGBestX, dGBestVal,
                nullptr,// traj
                false,//saveTraj
                N, iter, seed, states);//, best);
            cudaDeviceSynchronize();
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);

            float ms_iter=0;
            cudaEventElapsedTime(&ms_iter, t0, t1);
            cudaMemcpy(&hostGBestVal, dGBestVal, sizeof(double),        cudaMemcpyDeviceToHost);
            cudaMemcpy(hostGBestX.data(),  dGBestX,   DIM*sizeof(double), cudaMemcpyDeviceToHost);

            //printf("PSO‑Iter %2d execution time = %.3f ms   gBestVal = %.6e at [",iter, ms_iter, hostGBestVal);
            //for(int d=0; d<DIM; ++d) printf(" %.4f", hostGBestX[d]);
            //printf(" ]\n");
            ms_pso += ms_iter;
        }// end pso loop
        //printf("total pso time = %.3f\n", ms_pso+ms_init);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dX);
    cudaFree(dV);
    //cudaFree(dPBestX);
    cudaFree(dPBestVal);
    cudaFree(dGBestX);
    cudaFree(dGBestVal);
    return dPBestX;
}

template<int DIM>
Result<DIM> launch_reduction(int N, double* deviceResults,Result<DIM>* h_results) {
    // ArgMin & final print
    cub::KeyValuePair<int,double>* deviceArgMin;
    cudaMalloc(&deviceArgMin,     sizeof(*deviceArgMin));
    void*  d_temp_storage = nullptr;
    size_t temp_bytes      = 0;
    cub::DeviceReduce::ArgMin(
        d_temp_storage, temp_bytes,
        deviceResults, deviceArgMin, N);
    cudaMalloc(&d_temp_storage, temp_bytes);
    cub::DeviceReduce::ArgMin(
        d_temp_storage, temp_bytes,
        deviceResults, deviceArgMin, N);

    cub::KeyValuePair<int,double> h_argMin;
    cudaMemcpy(&h_argMin, deviceArgMin,
               sizeof(h_argMin),
               cudaMemcpyDeviceToHost);

    int    globalMinIndex = h_argMin.key;

    // print the “best” thread’s full record
    Result best = h_results[globalMinIndex];
    printf("Global best summary:\n");
    printf("   idx          = %d\n", best.idx);
    printf("   status       = %d\n", best.status);
    printf("   fval         = %.6f\n",best.fval);
    printf("   gradientNorm = %.6f\n",best.gradientNorm);
    printf("   iter         = %d\n",best.iter);
    printf("   coords       = [");
    for (int d = 0; d < DIM; ++d) {
         printf(" %.7f", best.coordinates[d]);
    }
    printf(" ]\n");

    cudaFree(deviceResults);
    cudaFree(deviceArgMin);
    cudaFree(d_temp_storage);
    return best;
}

template<typename Function, int DIM>
Result<DIM> launch_bfgs(const int N,const int pso_iter, const int MAX_ITER, const double upper, const double lower,double* pso_results_device,double* hostResults, double* deviceTrajectory, const int requiredConverged, const double tolerance, bool save_trajectories, float& ms_opt, std::string fun_name, curandState* states) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize,
        optimizeKernel<Function,DIM,128>,
        0, N);
    //printf("\nRecommended block size: %d\n", blockSize);
    
    // prepare optimizer buffers & copy hostResults --> device
    double* deviceResults;
    cudaMalloc(&deviceResults,    N * sizeof(double));
    cudaMemcpy(deviceResults, hostResults, N*sizeof(double), cudaMemcpyHostToDevice);

    dim3 optBlock(blockSize);
    dim3 optGrid((N + blockSize - 1) / blockSize);

    // optimizeKernel time
    cudaEvent_t startOpt, stopOpt;
    cudaEventCreate(&startOpt);
    cudaEventCreate(&stopOpt);
    cudaEventRecord(startOpt);

    Result<DIM>* h_results = new Result<DIM>[N];            // host copy
    Result<DIM>* d_results = nullptr;
    cudaMalloc(&d_results, N * sizeof(Result<DIM>));
    /*
    for(int i=0;i<DIM;i++){ 
        std::cout << hPBestX[i] << " ";
    }*/
    std::cout << std::endl;
    if (save_trajectories) {
        cudaMalloc(&deviceTrajectory, N*MAX_ITER*DIM*sizeof(double));
        optimizeKernel<Function,DIM,128>
            <<<optGrid,optBlock>>>(
                lower, upper,
                pso_results_device,
                deviceResults,
                deviceTrajectory,
                N,MAX_ITER,requiredConverged,tolerance,d_results,states,
                /*saveTraj=*/true);
    } else {
        optimizeKernel<Function,DIM,128>
            <<<optGrid,optBlock>>>(
                lower, upper,
                pso_results_device,
                deviceResults,
                /*traj=*/nullptr,
                N,MAX_ITER,requiredConverged,tolerance,d_results,states);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stopOpt);
    cudaEventSynchronize(stopOpt);
    cudaEventElapsedTime(&ms_opt, startOpt, stopOpt);
    //printf("\nOptimization Kernel execution time = %.3f ms\n", ms_opt);
    cudaEventDestroy(startOpt);
    cudaEventDestroy(stopOpt);

    cudaMemcpy(h_results, d_results, N * sizeof(Result<DIM>), cudaMemcpyDeviceToHost);

    //dump_data_2_file(h_results, fun_name, N, pso_iter);
    /*int countConverged = 0, surrender = 0, stopped = 0;
    for (int i = 0; i < N; ++i) {
        if (h_results[i].status == 1) { 
            countConverged++;
        } else if(h_results[i].status == 2) { // particle was stopped early
            stopped++;
        } else {
            surrender++;
        }
    }*/
    //printf("\n%d converged, %d stopped early, %d surrendered\n",countConverged, stopped, surrender);
    
    Result best = launch_reduction<DIM>(N, deviceResults, h_results);
    
    return best;
}

double calculate_euclidean_error(const std::string fun_name, const double* coordinates, const int dim) {
   double sum_sq = 0.0;
   if(fun_name == "rosenbrock") {
      for(int i=0;i<dim;i++) {
         double diff = coordinates[i] - 1.0;
         sum_sq += diff * diff;
      }
   } else if(fun_name == "rastrigin" || fun_name == "ackley") { // both rastrigin and ackley have the same coordinates for the global minimum
      for (int i = 0; i < dim; ++i) {
         sum_sq += coordinates[i] * coordinates[i];
      }
   }
   return std::sqrt(sum_sq);
}// end calculate_euclidean_error

inline curandState* initialize_states(int N, int seed, float& ms_rand) {
        // PRNG setup
        curandState* d_states;
        cudaMalloc(&d_states, N * sizeof(curandState));

        // Launch setup
        int threads = 256;
        int blocks  = (N + threads - 1) / threads;
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        util::setup_curand_states<<<blocks,threads>>>(d_states, seed, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_rand, t0, t1);
        cudaDeviceSynchronize();
        return d_states;
}

template<typename Function, int DIM>
Result<DIM> Zeus(const double lower,const double upper, double* hostResults,int N,int MAX_ITER, int PSO_ITER, int requiredConverged,std::string fun_name, double tolerance, const int seed)
{
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize,
        optimizeKernel<Function,DIM,128>,
        0, N);
    float ms_rand = 0.0f;
    curandState* states = initialize_states(N, seed, ms_rand);
    //printf("Recommended block size: %d\n", blockSize);
    bool save_trajectories = askUser2saveTrajectories();
    double* deviceTrajectory = nullptr;
    double* pso_results_device=nullptr;
    float ms_init = 0.0f, ms_pso = 0.0f; 
    if(PSO_ITER >= 0) {
        pso_results_device = launch_pso<Function, DIM>(PSO_ITER, N,lower, upper, ms_init,ms_pso, seed, states);
        //printf("pso init: %.2f main loop: %.2f", ms_init, ms_pso); 
    }// end if pso_iter > 0 
    if(!pso_results_device) 
       std::cout <<"still null" << std::endl;
    float ms_opt = 0.0f;
    Result best = launch_bfgs<Function, DIM>(N,PSO_ITER, MAX_ITER,upper, lower, pso_results_device, hostResults, deviceTrajectory, requiredConverged,tolerance, save_trajectories, ms_opt, fun_name, states);
    if(PSO_ITER > 0) { // optimzation routine is finished, so we can free that array on the device
         cudaFree(pso_results_device);
    } 

    double error = calculate_euclidean_error(fun_name, best.coordinates, DIM);
    append_results_2_tsv(DIM,N,fun_name,ms_init,ms_pso,ms_opt,ms_rand,MAX_ITER, PSO_ITER,error,best.fval, best.coordinates, best.idx, best.status, best.gradientNorm);
     
    cudaError_t cuda_error  = cudaGetLastError();
    if (cuda_error != cudaSuccess) { 
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
    }
    return best;
}// end Zeus

template<typename Function, int DIM>
void runOptimizationKernel(double lower, double upper, double* hostResults, int N, int MAX_ITER,int PSO_ITERS,int requiredConverged, std::string fun_name, double tolerance, int seed) {
//void runOptimizationKernel(double* hostResults, int N, int dim) {
    /*printf("first 20 hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }
    printf("\n");
    */
    Result best = Zeus<Function, DIM>(lower, upper, hostResults, N, MAX_ITER, PSO_ITERS, requiredConverged, fun_name,tolerance, seed);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //printf("Sorting the array with %d elements... ", N);  
    cudaEventRecord(start);
    //quickSort(hostResults, 0, N - 1);
    cudaEventRecord(stop);
    float milli = 0;
    cudaEventElapsedTime(&milli, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //printf("took %f ms\n",  milli);    

    /*printf("first 20 function values in hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }*/
    printf("\n");
//cudaMemGetInfo
}


template<int dim>
void selectAndRunOptimization(double lower, double upper,double* hostResults, int N, int MAX_ITER,int PSO_ITERS,int requiredConverged, double tolerance, int seed) {
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
    std::cout << " 7. Custom (user-defined objective via expression or kernel file)\n"
              << "Choice: ";
    std::cin >> choice;
    std::cin.ignore();

    switch(choice) {
        case 1:
            std::cout << "\n\n\tRosenbrock Function\n" << std::endl;
            runOptimizationKernel<util::Rosenbrock<dim>, dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged,"rosenbrock", tolerance, seed);
            break;
        case 2:
            std::cout << "\n\n\tRastrigin Function\n" << std::endl;
            runOptimizationKernel<util::Rastrigin<dim>, dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged, "rastrigin", tolerance, seed);
            break;
        case 3:
            std::cout << "\n\n\tAckley Function\n" << std::endl;
            runOptimizationKernel<util::Ackley<dim>, dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged, "ackley", tolerance, seed);
            break;
        case 4:
            if constexpr (dim != 2) {
                std::cerr << "Error: GoldsteinPrice is defined for 2 dimensions only.\n";
            } else {
                std::cout << "\n\n\tGoldsteinPrice Function\n" << std::endl;
                runOptimizationKernel<util::GoldsteinPrice<dim>, dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged, "goldstein", tolerance, seed);
            }
            break;
        case 5:
            if constexpr (dim != 2) {
                std::cerr << "Error: Eggholder is defined for 2 dimensions only.\n";
            } else {
                std::cout << "\n\n\tEggholder Function\n" << std::endl;
                runOptimizationKernel<util::Eggholder<dim>, dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged, "eggholder", tolerance, seed);
            }
            break;
        case 6:
            if constexpr (dim != 2) {
                std::cerr << "Error: Himmelblau is defined for 2 dimensions only.\n";
            } else {
                std::cout << "\n\n\tHimmelblau Function\n" << std::endl;
                runOptimizationKernel<util::Himmelblau<dim>, dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged, "himmelblau", tolerance, seed);
            }
            break;
        case 7:
            std::cout << "\n\n\tCustom User-Defined Function\n" << std::endl;
            // for a more complex custom function, one option is to let the user provide a path
            // to a cuda file and compile it at runtime. 
            //runOptimizationKernel<UserDefined<dim>, dim>(lower, upper, hostResults, hostIndices,
            //                                             hostCoordinates, N, MAX_ITER);
            break;
        default:
            std::cerr << "Invalid selection!\n";
            exit(1);
    }
}

//#ifndef UNIT_TEST
//#ifndef NO_MAIN
#if !defined(UNIT_TEST) && !defined(TABLE_GEN)
int main(int argc, char* argv[]) {
    printf("Production main() running\n");
    if (argc != 9) {
	 std::cerr << "Usage: " << argv[0] << " <lower_bound> <upper_bound> <max_iter> <pso_iters> <converged> <number_of_optimizations> <tolerance> <seed>\n";
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

    
    std::cout << "Tolerance: " << std::setprecision(10) << tolerance << "\nseed: " << seed <<"\n";

    //const size_t N = 128*4;//1024*128*16;//pow(10,5.5);//128*1024*3;//*1024*128;
    const int dim = 10;
    double hostResults[N];// = new double[N];
    std::cout << "number of optimizations = " << N << " max_iter = " << MAX_ITER << " dim = " << dim << std::endl;
     
    double f0 = 333777; // initial function value

    // logic to set the stact size limit to 65 kB per thread 
    size_t currentStackSize = 0;
    cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
    printf("Current stack size: %zu bytes\n", currentStackSize);
    size_t newStackSize = 64  * 1024; // 65 kB
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (err != cudaSuccess) {
        printf("cudaDeviceSetLimit error: %s\n", cudaGetErrorString(err));
        return 1;
    } else {
            printf("Successfully set stack size to %zu bytes\n", newStackSize);
    }// end stack size limit

    char cont = 'y';
    while (cont == 'y' || cont == 'Y') {
        for (int i = 0; i < N; i++) {
            hostResults[i] = f0;
        }
        selectAndRunOptimization<dim>(lower, upper, hostResults, N, MAX_ITER,PSO_ITERS, requiredConverged, tolerance, seed);
        std::cout << "\nDo you want to optimize another function? (y/n): ";
        std::cin >> cont;
        std::cin.ignore();
    }
    

    //for(int i=0; i<N; i++) {
    //    hostResults[i] = f0;
    //}
    //selectAndRunOptimization<dim>(lower, upper, hostResults, hostIndices, hostCoordinates, N, MAX_ITER);
    return 0;
}
#endif
