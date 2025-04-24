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

#include <filesystem>

namespace dual {

class DualNumber {
public:
    double real;
    double dual;

    __host__ __device__ DualNumber(double real = 0.0, double dual = 0.0) : real(real), dual(dual) {}

    __host__ __device__ DualNumber& operator+=(const DualNumber& rhs) {
        real += rhs.real;
        dual += rhs.dual;
        return *this;
    }

    __host__ __device__ DualNumber operator+(const DualNumber& rhs) const {
        return DualNumber(real + rhs.real, dual + rhs.dual);
    }

    __host__ __device__ DualNumber operator-(const DualNumber& rhs) const {
        return DualNumber(real - rhs.real, dual - rhs.dual);
    }

    __host__ __device__ DualNumber operator*(const DualNumber& rhs) const {
        return DualNumber(real * rhs.real, dual * rhs.real + real * rhs.dual);
    }

    __host__ __device__ DualNumber operator/(const DualNumber& rhs) const {
        double denom = rhs.real * rhs.real;
        return DualNumber(real / rhs.real, (dual * rhs.real - real * rhs.dual) / denom);
    }
    // operator for double - DualNumber
    __host__ __device__ friend DualNumber operator-(double lhs, const DualNumber& rhs) {
        return DualNumber(lhs - rhs.real, -rhs.dual);
    }

    // operator for double * DualNumber
    __host__ __device__ friend DualNumber operator*(double lhs, const DualNumber& rhs) {
        return DualNumber(lhs * rhs.real, lhs * rhs.dual);
    }
};

__host__ __device__ inline dual::DualNumber dual_abs(const dual::DualNumber &a) {
    return (a.real < 0.0) ? dual::DualNumber(-a.real, -a.dual) : a;
}

__host__ __device__ DualNumber sin(const DualNumber& x) {
    return DualNumber(sinf(x.real), x.dual * cosf(x.real));
}

__host__ __device__ DualNumber cos(const DualNumber& x) {
    return DualNumber(cosf(x.real), -x.dual * sinf(x.real));
}

__host__ __device__ DualNumber exp(const DualNumber& x) {
    double ex = expf(x.real);
    return DualNumber(ex, x.dual * ex);
}

__host__ __device__ DualNumber sqrt(const DualNumber& x) {
    double sr = sqrtf(x.real);
    return DualNumber(sr, x.dual / (2.0 * sr));
}

__host__ __device__ DualNumber atan2(const DualNumber& y, const DualNumber& x) {
    double denom = x.real * x.real + y.real * y.real;
    return DualNumber(atan2f(y.real, x.real), (x.real * y.dual - y.real * x.dual) / denom);
}

template<typename T>
__host__ __device__ T pow(const T& base, double exponent) {
    return T(powf(base.real, exponent), exponent * powf(base.real, exponent - 1) * base.dual);
}

} // end of dual



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


template<int DIM>
__device__ dual::DualNumber rosenbrock(const dual::DualNumber* x) {
    dual::DualNumber sum = 0.0;
    for (int i = 0; i < DIM - 1; ++i) {
	sum += dual::pow(1 - x[i],2) + 100 * dual::pow(x[i+1] - dual::pow(x[i], 2), 2); 
        //sum = sum + 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
    }
    return sum;
}

template<int DIM>
__device__ double rosenbrock(const double* x) {
    double sum = 0.0;
    for (int i = 0; i < DIM - 1; ++i) {
        sum += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
    }
    return sum;
}

template<int DIM>
__device__ dual::DualNumber rastrigin(const dual::DualNumber* x) {
    const double A = 10.0;
    dual::DualNumber sum(A * DIM, 0.0);
    for(int i=0; i<DIM; i++){
	dual::DualNumber xi_sq = x[i] * x[i];
	dual::DualNumber cterm = cos(2.0 * M_PI * x[i]);
        sum = sum + (xi_sq - A * cterm);
    }
    return sum;
    /*dual::DualNumber sum = 10 * DIM;
    for (int i = 0; i < DIM; ++i) {
        sum = sum + x[i] * x[i] - 10 * cos(2 * M_PI * x[i].real);
    }
    return sum;*/
}

template<int DIM>
__device__ double rastrigin(const double* x) {
    const double A = 10.0;
    double val = A * DIM;
    for(int i=0; i<DIM; i++){
        double xi = x[i];
        val += xi*xi - A*::cosf(2.0f*M_PI*xi);
    }
    return val;
    /*
    	double sum = 10 * DIM;
    for (int i = 0; i < DIM; ++i) {
        sum += x[i] * x[i] - 10 * cos(2 * M_PI * x[i]);
    }
    return sum;
*/}


template<int DIM>
struct Rosenbrock {
    __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        //printf("in dualrosen");
        return rosenbrock<DIM>(x);
    }

    __device__ static double evaluate(const double* x) {
        return rosenbrock<DIM>(x);
    }
};

template<int DIM>
struct Rastrigin {
    __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return rastrigin<DIM>(x);
    }

    __device__ static double evaluate(const double* x) {
        return rastrigin<DIM>(x);
    }
};

// Ackley Function (general d-dimensions)
//   f(x) = -20 exp\Bigl(-0.2\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}\Bigr)
//          - exp\Bigl(\frac{1}{d}\sum_{i=1}^{d}\cos(2\pi x_i)\Bigr)
//          + 20 + e
template<int DIM>
__device__ dual::DualNumber ackley(const dual::DualNumber* x) {
    dual::DualNumber sum_sq = 0.0;
    dual::DualNumber sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum_sq += dual::pow(x[i], 2);
        sum_cos += dual::cos(2.0 * M_PI * x[i]);
    }
    dual::DualNumber term1 = dual::DualNumber(-20.0) * dual::exp(-0.2 * dual::sqrt(sum_sq / DIM));
    dual::DualNumber term2 = dual::DualNumber(0.0) - dual::exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + dual::exp(1.0);
}

template<int DIM>
__device__ double ackley(const double* x) {
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum_sq += x[i] * x[i];
        sum_cos += cos(2.0 * M_PI * x[i]);
    }
    double term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / DIM));
    double term2 = -exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + exp(1.0);
}

template<int DIM>
struct Ackley {
    __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return ackley<DIM>(x);
    }
    __device__ static double evaluate(const double* x) {
        return ackley<DIM>(x);
    }
};

// Goldstein-Price Function
//   f(x,y) = [1+(x+y+1)^2 (19-14x+3x^2-14y+6xy+3y^2)]
//            [30+(2x-3y)^2 (18-32x+12x^2+48y-36xy+27y^2)]
template<int DIM>
__device__ dual::DualNumber goldstein_price(const dual::DualNumber* x) {
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
__device__ double goldstein_price(const double* x) {
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
    __device__ static double evaluate(const double* x) {
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
    __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return eggholder<DIM>(x);
    }
    __device__ static double evaluate(const double* x) {
        return eggholder<DIM>(x);
    }
};


// Himmelblau's Function (2D only)
template<int DIM>
__device__ dual::DualNumber himmelblau(const dual::DualNumber* x) {
    static_assert(DIM == 2, "Himmelblau's function is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0], x2 = x[1];
    dual::DualNumber term1 = dual::pow(x1 * x1 + x2 - dual::DualNumber(11.0), 2);
    dual::DualNumber term2 = dual::pow(x1 + x2 * x2 - dual::DualNumber(7.0), 2);
    return term1 + term2;
}

template<int DIM>
__device__ double himmelblau(const double* x) {
    static_assert(DIM == 2, "Himmelblau's function is defined for 2 dimensions only.");
    double x1 = x[0], x2 = x[1];
    double term1 = pow(x1 * x1 + x2 - 11.0, 2);
    double term2 = pow(x1 + x2 * x2 - 7.0, 2);
    return term1 + term2;
}

template<int DIM>
struct Himmelblau {
    __device__ static dual::DualNumber evaluate(const dual::DualNumber* x) {
        return himmelblau<DIM>(x);
    }
    __device__ static double evaluate(const double* x) {
        return himmelblau<DIM>(x);
    }
};


template<typename Function, int DIM>
__device__ void calculateGradientUsingAD(double *x, double *gradient) {
    dual::DualNumber xDual[DIM];

    for (int i = 0; i < DIM; ++i) { // // iterate through each dimension (vairbale)
        xDual[i] = dual::DualNumber(x[i], 0.0);
    }

    // calculate the partial derivative of  each dimension
    for (int i = 0; i < DIM; ++i) {
        xDual[i].dual = 1.0; // derivative w.r.t. dimension i
        dual::DualNumber result = Function::evaluate(xDual); // evaluate the function using AD
        gradient[i] = result.dual; // store derivative
        //printf("\nxDual[%d]: %f, grad[%d]: %f ",i,xDual[i].real,i,gradient[i]);
        xDual[i].dual = 0.0;
    }
}

__device__ double generate_random_double(unsigned int seed, double lower, double upper)
{
    curandState state;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state); // initialize cuRAND with unique sequence number
    return lower + (upper + (-lower)) * curand_uniform_double(&state);
    //return -5.0 + (-3.0 + 5.12) * curand_uniform_double(&state); // return scaled double
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

//const int MAX_ITER = 64;

template<typename Function, int DIM, unsigned int blockSize>
__global__ void optimizeKernel(double lower, double upper, double* deviceResults, int* deviceIndices, double* deviceCoordinates, double* deviceTrajectory, int N, int MAX_ITER, bool save_trajectories = false) {
//template<typename Function, int DIM, unsigned int blockSize> 
//__global__ void optimizeKernel(double* devicePoints,double* deviceResults, int N) {
//__global__ void optimizeKernel(double lower, double upper, double* deviceResults, int* deviceIndices, double* deviceCoordinates, int N, int MAX_ITER) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    //int early_stopping = 0;
    double H[DIM * DIM];
    double g[DIM], x[DIM], x_new[DIM], p[DIM], g_new[DIM], delta_x[DIM], delta_g[DIM];//, new_direction[DIM];
    double tolerance = 1e-8;
    // Line Search params

    util::initialize_identity_matrix(H, DIM);
    unsigned int seed = 135;
    //unsigned int seed = 246; 
    //unsigned int seed = 37;
    
    int num_steps = 0;

    //unsigned int seed = 1;
    //unsigned int seed = 456;
    for (int i = 0; i < DIM; ++i) {
        x[i] = util::generate_random_double(seed+idx*DIM+i, lower, upper);// devicePoints[i * dim + idx];
        //deviceStartingPositions[idx * DIM + i] = x[i];
        //printf("x[%d]: %f",i,x[i]);//<<std::endl;
    }
    double f0 = Function::evaluate(x);//rosenbrock_device(x, DIM);
    deviceResults[idx] = f0;
    double bestVal = f0;
    if (idx == 0) {
       printf("\n\nf0 = %f", f0);
    }
    util::calculateGradientUsingAD<Function, DIM>(x, g);
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        num_steps++;
	//printf("idx = %d", idx);
	//printf("x[0] = %f", x[0]);
	//printf("\n\n\nit#%d",iter);
	//Function::gradient(x, g, DIM);
	//rosenbrock_gradient_device(x, g, DIM);
	//util::calculateGradientUsingAD<Function, DIM>(x, g);
        
	double grad_norm = 0.0;
	for (int i = 0; i < DIM; ++i) { grad_norm += g[i] * g[i];}
        grad_norm = sqrt(grad_norm);
        if (grad_norm < tolerance) {
            //printf("converged");
	    break; //converged
	}
       	//else {
	//    printf("norm of gradient: %f", grad_norm);
	//}

	//d0 = 0.0;
        // compute the search direction p = -H * g
        for (int i = 0; i < DIM; i++) {
            double sum=0.0;
	    for (int j = 0; j < DIM; j++) {
                sum += H[i * DIM + j] * g[j]; // i * dim + j since H is flattened arr[]
            }    
	    p[i] = -sum;
        }

	// use the alpha obtained from the line search
	double alpha = util::line_search<Function,DIM>(bestVal, x, p, g);
	if(alpha == 0.0) {
            printf("Alpha is zero, no movement in iteration=%d\n", iter);
            alpha = 1e-3; 
        }
        //printf("alpha before updat at it%de: %f", iter, alpha);

	// update current point x by taking a step size of alpha in the direction p
	for (int i = 0; i < DIM; ++i) {
            x_new[i] = x[i] + alpha * p[i];
	    delta_x[i] = x_new[i] - x[i];
	    //printf("\nnewx[%d]: %f",i,x_new[i]);
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
	
	//  deviceTrajectory layout: idx * (MAX_ITER * DIM) + iter * DIM + i
	if (save_trajectories) {
	   for (int i = 0; i < DIM; i++) {
               deviceTrajectory[idx * (MAX_ITER * DIM) + iter * DIM + i] = x[i];
           }
	} 

	//for(int i=0; i<DIM; ++i) {x[i] = x_new[i];}
    }// end outer for
    bestVal = Function::evaluate(x);//rosenbrock_device(x, DIM);
    //printf("\nmax iterations reached, predicted minima = %f\n", minima);
    deviceResults[idx] = bestVal;
    deviceIndices[idx] = idx;
    for (int i = 0; i < DIM; ++i) {
        deviceCoordinates[idx * DIM + i] = x[i];
    }
}// end optimizerKernel

const int doublesPerThread = 1024;

__global__ void setup_states(curandState *state, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        curand_init(1234, id, 0, &state[id]);
    }
}

__global__ void generate_random_doubles(curandState *state, double *out, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int startIdx = id * doublesPerThread;
    if (startIdx < n) {
        for (int i = 0; i < doublesPerThread && (startIdx + i) < n; i++) {    
            out[startIdx + i] = -5 + 2 * curand_uniform_double(&state[id]);
        }
    }
}

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

template<typename Function, int DIM>
cudaError_t launchOptimizeKernel(double lower, double upper, double* hostResults, int* hostIndices, double* hostCoordinates, int N, int MAX_ITER, std::string fun_name) {
    int blokSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blokSize, optimizeKernel<Function, DIM, 128>, 0, 0);
    printf("Recommended block size: %d\n", blokSize);    
    bool save_trajectories = askUser2saveTrajectories();
	    
    double* deviceTrajectory = nullptr;
    

    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //printf("");
    double* deviceResults;
    int* deviceIndices;
    double* deviceCoordinates;
    cub::KeyValuePair<int, double>* deviceArgMin; // For ArgMin result

    cudaMalloc(&deviceResults, N * sizeof(double));
    cudaMalloc(&deviceIndices, N * sizeof(int));
    cudaMalloc(&deviceCoordinates, N * DIM * sizeof(double));
    cudaMalloc(&deviceArgMin, sizeof(cub::KeyValuePair<int, double>));
    cudaMemcpy(deviceResults, hostResults, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    //optimizeKernel<Function, DIM, 128><<<numBlocks, blockSize>>>(
    //lower, upper, deviceResults, deviceIndices, deviceCoordinates, deviceTrajectory, N, MAX_ITER);
    if (save_trajectories) {// only allocate large trajectory array if user wants to!!
	cudaMalloc(&deviceTrajectory, N * MAX_ITER * DIM * sizeof(double));
        optimizeKernel<Function, DIM, 256><<<numBlocks, blockSize>>>(lower, upper,deviceResults, deviceIndices, deviceCoordinates, deviceTrajectory, N, MAX_ITER, save_trajectories);
    } else {
        optimizeKernel<Function, DIM, 256><<<numBlocks, blockSize>>>(lower, upper,deviceResults, deviceIndices, deviceCoordinates, deviceTrajectory, N, MAX_ITER);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliKernel = 0;
    cudaEventElapsedTime(&milliKernel, start, stop);
    printf("\nOptimization Kernel execution time = %f ms\n", milliKernel);

    // Determine temporary device storage requirements for ArgMin
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, deviceResults, deviceArgMin, N);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run ArgMin reduction
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, deviceResults, deviceArgMin, N);

    // Retrieve the ArgMin result
    cub::KeyValuePair<int, double> h_argMin;
    cudaMemcpy(&h_argMin, deviceArgMin, sizeof(cub::KeyValuePair<int, double>), cudaMemcpyDeviceToHost);

    int globalMinIndex = h_argMin.key;
    double globalMin = h_argMin.value;

    printf("\nGlobal Minima: %f\n", globalMin);
    printf("Global Minima Index: %d\n", globalMinIndex);

    // Copy device coordinates for the global minimum index to host
    cudaMemcpy(hostCoordinates, deviceCoordinates + globalMinIndex * DIM, DIM * sizeof(double), cudaMemcpyDeviceToHost);
        printf("Coordinates associated with Global Minima:\n");
    for (int i = 0; i < DIM; ++i) {
        printf("x[%d] = %f\n", i, hostCoordinates[i]);
    }
    /*
    // Compute Euclidean error based on known optimum
    double error = 0.0;
    if (fun_name == "rosenbrock") {
        for (int i = 0; i < DIM; i++) {
            double diff = hostCoordinates[i] - 1.0;
            error += diff * diff;
        }
    } else if (fun_name == "rastrigin" || fun_name == "ackley") {
        for (int i = 0; i < DIM; i++) {
            error += hostCoordinates[i] * hostCoordinates[i];
        }
    } else if (fun_name == "goldstein") {
        double optimum[DIM] = {0.0, -1.0};
        for (int i = 0; i < DIM; i++) {
            double diff = hostCoordinates[i] - optimum[i];
            error += diff * diff;
        }
    } else if (fun_name == "himmelblau") {
        const int numCandidates = 4;
        double candidates[numCandidates][2] = {
        { 3.0,  2.0},
        {-2.805118,  3.131312},
        {-3.779310, -3.283186},
        { 3.584428, -1.848126}
        };
        double minErrorSquared = std::numeric_limits<double>::max();
        for (int c = 0; c < numCandidates; c++) {
            double errorCandidate = 0.0;
            for (int i = 0; i < 2; i++) {
                double diff = hostCoordinates[i] - candidates[c][i];
                errorCandidate += diff * diff;
            } // end inner for
            if (errorCandidate < minErrorSquared)
                minErrorSquared = errorCandidate;
        }// end outer for
        error = minErrorSquared;
    } else {
        error = std::numeric_limits<double>::quiet_NaN();
    }
    error = sqrt(error);
    */
    double error;
    double fStar = 0.0;                         // f(x*) for all five functions
    double fVal = globalMin;  // your routine

    if (std::abs(fStar) > 0.0)
        error = std::abs(fVal - fStar) / std::abs(fStar);
    else
        error = std::abs(fVal); 
    // Append results to a .tsv file in precision 17
    {
        std::string filename = std::to_string(DIM) + "d_results.tsv";
        std::ofstream outfile(filename, std::ios::app);
        // Convert kernel time from milliseconds to seconds.
        double time_seconds = milliKernel / 1000.0;
        outfile << fun_name << "\t" << N << "\t" 
            << std::fixed << std::setprecision(17) << time_seconds << "\t"
            << error << "\t"
            << globalMin << "\t";
        for (int i = 0; i < DIM; i++) {
            outfile << hostCoordinates[i];
            if (i < DIM - 1)
                outfile << ",";
        }
        outfile << "\n";
        outfile.close();
        printf("results are saved to %s", filename.c_str());
    }

    if (save_trajectories) {
        double* hostTrajectory = new double[N * MAX_ITER * DIM];
        cudaMemcpy(hostTrajectory, deviceTrajectory, N * MAX_ITER * DIM * sizeof(double), cudaMemcpyDeviceToHost);
        writeTrajectoryData(hostTrajectory, N, MAX_ITER, DIM, fun_name, "./data");
    
        delete[] hostTrajectory;
        cudaFree(deviceTrajectory);
    } //end save trajectories
    
    //hostResults[0] = globalMin;
    //cudaMemcpy(hostResults + 1, deviceResults + 1, (N - 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostResults, deviceResults, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deviceResults);
    cudaFree(deviceIndices);
    cudaFree(deviceCoordinates);
    cudaFree(deviceArgMin);
    cudaFree(d_temp_storage);

    return cudaSuccess;
}


void swap(double* a, double* b) {
    double t = *a;
    *a = *b;
    *b = t;
}
int partition(double arr[], int low, int high) {
    double pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quickSort(double arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

template<typename Function, int DIM>
void runOptimizationKernel(double lower, double upper, double* hostResults, int* hostIndices, double* hostCoordinates, int N, int MAX_ITER, std::string fun_name) {
//void runOptimizationKernel(double* hostResults, int N, int dim) {
    /*printf("first 20 hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }
    printf("\n");
    */
    cudaError_t error = launchOptimizeKernel<Function, DIM>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER, fun_name);
    if (error != cudaSuccess) {
        printf("CUDA error: %s", cudaGetErrorString(error));
    } else {
        printf("\nSuccess!! No Error!\n");
    }
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

struct Parameters {
    double lower;
    double upper;
    size_t N;
    int max_iter;
    int dim;
};

Parameters parseArguments(int argc, char* argv[]) {
    Parameters params;
    bool valid = false;
    if (argc == 6) {
        try {
            params.lower    = std::stod(argv[1]);
            params.upper    = std::stod(argv[2]);
            params.N        = static_cast<size_t>(std::stoul(argv[3]));
            params.max_iter = std::stoi(argv[4]);
            params.dim      = std::stoi(argv[5]);
            valid = true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error converting command-line args: " << e.what() << "\n";
        }
    } else {
        std::cerr << "Incorrect number of command-line arguments.\n";
    }

    while (!valid) {
        std::string input_line;
        std::cout << "Enter lower_bound upper_bound opt_count steps2take dimension: ";
        std::getline(std::cin, input_line);
        std::istringstream iss(input_line);
        if (!(iss >> params.lower >> params.upper >> params.N >> params.max_iter >> params.dim)) {
            std::cerr << "Invalid input format.\n";
            std::string answer;
            std::cout << "Try again? ";
            std::getline(std::cin, answer);
            if (!(answer == "Y" || answer == "y" || answer == "Yes" || answer == "yes"))
                std::exit(1);
        } else {
            valid = true;
        }
    }
    return params;
}

template<int dim>
void selectAndRunOptimization(double lower, double upper,
                              double* hostResults, int* hostIndices,
                              double* hostCoordinates, int N, int MAX_ITER) {
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
            runOptimizationKernel<util::Rosenbrock<dim>, dim>(lower, upper, hostResults, hostIndices,hostCoordinates, N, MAX_ITER, "rosenbrock");
            break;
        case 2:
            std::cout << "\n\n\tRastrigin Function\n" << std::endl;
            runOptimizationKernel<util::Rastrigin<dim>, dim>(lower, upper, hostResults, hostIndices,hostCoordinates, N, MAX_ITER, "rastrigin");
            break;
        case 3:
            std::cout << "\n\n\tAckley Function\n" << std::endl;
            runOptimizationKernel<util::Ackley<dim>, dim>(lower, upper, hostResults, hostIndices,hostCoordinates, N, MAX_ITER, "ackley");
            break;
        case 4:
            if constexpr (dim != 2) {
                std::cerr << "Error: GoldsteinPrice is defined for 2 dimensions only.\n";
            } else {
                std::cout << "\n\n\tGoldsteinPrice Function\n" << std::endl;
                runOptimizationKernel<util::GoldsteinPrice<dim>, dim>(lower, upper, hostResults, hostIndices,hostCoordinates, N, MAX_ITER, "goldstein");
            }
            break;
        case 5:
            if constexpr (dim != 2) {
                std::cerr << "Error: Eggholder is defined for 2 dimensions only.\n";
            } else {
                std::cout << "\n\n\tEggholder Function\n" << std::endl;
                runOptimizationKernel<util::Eggholder<dim>, dim>(lower, upper, hostResults, hostIndices, hostCoordinates, N, MAX_ITER, "eggholder");
            }
            break;
        case 6:
            if constexpr (dim != 2) {
                std::cerr << "Error: Himmelblau is defined for 2 dimensions only.\n";
            } else {
                std::cout << "\n\n\tHimmelblau Function\n" << std::endl;
                runOptimizationKernel<util::Himmelblau<dim>, dim>(lower, upper, hostResults, hostIndices,hostCoordinates, N, MAX_ITER, "himmelblau");
            }
            break;
        case 7:
            std::cout << "\n\n\tCustom User-Defined Function\n" << std::endl;
            // For a more complex custom function, one option is to let the user provide a path
            // to a .cu file and compile it at runtime. 
            //runOptimizationKernel<UserDefined<dim>, dim>(lower, upper, hostResults, hostIndices,
            //                                             hostCoordinates, N, MAX_ITER);
            break;
        default:
            std::cerr << "Invalid selection!\n";
            exit(1);
    }
}

#ifndef UNIT_TEST
int main(int argc, char* argv[]) {
    printf("Production main() running\n");
    if (argc != 5) {
	 std::cerr << "Usage: " << argv[0] << " <lower_bound> <upper_bound> <max_iter> <number_of_optimizations\n";
        return 1;
    }
    double lower = std::atof(argv[1]);
    double upper = std::atof(argv[2]);   	

    int MAX_ITER = std::stoi(argv[3]);
    int N = std::stoi(argv[4]);
    

    /* Use parsed parameters.
    std::cout << "Parsed Values:\n"
              << "lower: " << params.lower << "\n"
              << "upper: " << params.upper << "\n"
              << "N: " << params.N << "\n"
              << "max_iter: " << params.max_iter << "\n"
              << "dim: " << params.dim << "\n";
    */
    //const size_t N = 128*4;//1024*128*16;//pow(10,5.5);//128*1024*3;//*1024*128;
    const int dim =2;
    double hostResults[N];// = new double[N];
    std::cout << "number of optimizations = " << N << " max_iter = " << MAX_ITER << " dim = " << dim << std::endl;

     
    int hostIndices[N];
    double hostCoordinates[dim];
    double f0 = 333777; // initial function value


    size_t currentStackSize = 0;
    cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
    printf("Current stack size: %zu bytes\n", currentStackSize);

    size_t newStackSize = 64  * 1024; // 65 kB per thread
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (err != cudaSuccess) {
        printf("cudaDeviceSetLimit error: %s\n", cudaGetErrorString(err));
        return 1;
    } else {
            printf("Successfully set stack size to %zu bytes\n", newStackSize);
    }


    char cont = 'y';
    while (cont == 'y' || cont == 'Y') {
        for (int i = 0; i < N; i++) {
            hostResults[i] = f0;
        }
        selectAndRunOptimization<dim>(lower, upper, hostResults, hostIndices, hostCoordinates, N, MAX_ITER);
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
