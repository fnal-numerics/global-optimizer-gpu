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


__device__ void vector_add(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

__device__ void vector_scale(const double* a, double scalar, double* result, int dim) {
    for (int i = 0; i < dim; ++i) {
        result[i] = a[i] * scalar;
    }
}

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

/*
__device__ double rosenbrock_device(double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        sum += 100 * pow2(x[i + 1] - pow2(x[i])) + pow2(1-x[i]);
    }

    return sum;
}

__device__ void rosenbrock_gradient_device(double* x, double* grad, int n) {
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - pow2(x[0]));
    for (size_t i = 1; i < n - 1; ++i) {
        grad[i] = -2 * (1 - x[i]) + 200 * (x[i] - pow2(x[i - 1])) - 400 * x[i] * (x[i + 1] - pow2(x[i]));
    }
    grad[n - 1] = 200 * (x[n - 1] - pow2(x[n - 2]));
}

__device__ double rastrigin_device(double* x, int n) {
    double sum = 10 * n;
    for (int i = 0; i < n; ++i) {
        sum += (x[i] * x[i] - 10 * cos(2 * M_PI * x[i]));
    }
    return sum;
}
*/

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
__device__ void bfgs_update(double* H, const double* s, const double* y, double sTy){
    if(::fabs(sTy)<1e-14) return;
    double rho = 1.0/sTy;
    // s y^T
    double sY[DIM*DIM], yS[DIM*DIM], sS[DIM*DIM];
    //outer_product_device<DIM>(s,y,sY);
    //outer_product_device<DIM>(y,s,yS);
    //outer_product_device<DIM>(s,s,sS);
    outer_product_device(s, y, sY, DIM);
    outer_product_device(y, s, yS, DIM);
    outer_product_device(s, s, sS, DIM);
    for(int i=0;i<DIM;i++){
        for(int j=0;j<DIM;j++){
            double Iij = (i==j)?1.0:0.0;
            int idx = i*DIM + j;
            // left multiply with (I - rho s y^T)
            double factor1 = Iij - rho*sY[idx];
            // multiply factor1 with row i of H
            double rowMult=0.0;
            for(int k=0;k<DIM;k++){
                rowMult += factor1 * H[i*DIM + k];
            }
            // right multiply (I - rho y s^T)
            double factor2=0.0;
            for(int m=0;m<DIM;m++){
                double term = ( (m==j)?1.0:0.0 ) - rho*yS[m*DIM + j];
                factor2 += rowMult * term;
            }
            // + rho * s s^T
            double addTerm = rho*sS[idx];
            double val = factor2 + addTerm;
            H[idx] = val;
        }
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

// Goldstein-Price Function (2D only)
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

// Eggholder Function (2D only)
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
__global__ void optimizeKernel(double lower, double upper, double* deviceResults, int* deviceIndices, double* deviceCoordinates, double* deviceTrajectory, int N, int MAX_ITER) {
//template<typename Function, int DIM, unsigned int blockSize> 
//__global__ void optimizeKernel(double* devicePoints,double* deviceResults, int N) {
//__global__ void optimizeKernel(double* deviceResults, int* deviceIndices, double* deviceCoordinates, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    //int early_stopping = 0;
    double H[DIM * DIM];
    double g[DIM], x[DIM], x_new[DIM], p[DIM], g_new[DIM], delta_x[DIM], delta_g[DIM];//, new_direction[DIM];
    /*for (int i = 0; i < DIM; i++) H[i] = 0;
    for (int i = 0; i < DIM; i++) {
        g[i] = 0;
        x[i] = 0;
        x_new[i] = 0;
        p[i] = 0;
        g_new[i] = 0;
        delta_x[i] = 0;
        delta_g[i] = 0;
    }*/
    //double s[DIM];
    //double minima = 10000000;
    //double min_alpha = 0.3;
    //double max_alpha = 1.0;
    //double delta_dot = 0;
    double tolerance = 1e-8;
    // Line Search params
    //double min_f = -1000.0;
    //int max_iter = 5;
    //double lambda = 15.0;
    //double alpha = min_alpha;
    //double d0 = 0;

    util::initialize_identity_matrix(H, DIM);
    //unsigned int seed = 135;
    //unsigned int seed = 246; 
    
    int num_steps = 0;

    unsigned int seed = 379;
    //unsigned int seed = 1234;
    for (int i = 0; i < DIM; ++i) {
        x[i] = util::generate_random_double(seed+idx*DIM+i, lower, upper);// devicePoints[i * dim + idx];
        //deviceStartingPositions[idx * DIM + i] = x[i];
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
            printf("converged");
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
	
        // deviceTrajectory layout: idx * (MAX_ITER * DIM) + iter * DIM + i
	for (int i = 0; i < DIM; i++) {
            deviceTrajectory[idx * (MAX_ITER * DIM) + iter * DIM + i] = x[i];
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


template<typename Function, int DIM>
cudaError_t launchOptimizeKernel(double lower, double upper, double* hostResults, int* hostIndices, double* hostCoordinates, int N, int MAX_ITER) {
    int blokSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blokSize, optimizeKernel<Function, DIM, 128>, 0, 0);
    printf("Recommended block size: %d\n", blokSize);    

    double* deviceTrajectory;
    cudaMalloc(&deviceTrajectory, N * MAX_ITER * DIM * sizeof(double));

    dim3 blockSize(128);
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
    
    optimizeKernel<Function, DIM, 128><<<numBlocks, blockSize>>>(
    lower, upper, deviceResults, deviceIndices, deviceCoordinates, deviceTrajectory, N, MAX_ITER);
    //optimizeKernel<Function, DIM, 128><<<numBlocks, blockSize>>>(deviceResults, deviceIndices, deviceCoordinates, N);

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
    /* Allocate host memory for new arrays
    double* hostStartingPositions = new double[N * DIM];
    double* hostStoppingPositions = new double[N * DIM];
    double* hostFnValueStart = new double[N];
    double* hostFnValueStop = new double[N];
    int* hostNumSteps = new int[N];
    */

    double* hostTrajectory = new double[N * MAX_ITER * DIM];
    cudaMemcpy(hostTrajectory, deviceTrajectory, N * MAX_ITER * DIM * sizeof(double), cudaMemcpyDeviceToHost);

    /* Copy data from device to host
    cudaMemcpy(hostStartingPositions, deviceStartingPositions, N * DIM * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostStoppingPositions, deviceStoppingPositions, N * DIM * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostFnValueStart, deviceFnValueStart, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostFnValueStop, deviceFnValueStop, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostNumSteps, deviceNumSteps, N * sizeof(int), cudaMemcpyDeviceToHost);
    */

    // Write data to file
    // Format:
    // Optimization_Index Step X_0 X_1 ... X_(DIM-1)
    std::string filename = std::to_string(MAX_ITER*N)+"/trajectories/"+std::to_string(MAX_ITER)+"it_"+ std::to_string(N) + "opt.txt"; 
    std::ofstream stepOut(filename);
    stepOut << "OptIndex Step";
    for (int d = 0; d < DIM; d++) {
        stepOut << " X_" << d;
    }
    stepOut << "\n";

    for (int i = 0; i < N; i++) {
        for (int it = 0; it < MAX_ITER; it++) {
            stepOut << i << " " << it << " ";
            for (int d = 0; d < DIM; d++) {
                stepOut << hostTrajectory[i * (MAX_ITER * DIM) + it * DIM + d] << (d == DIM-1 ? "\n" : " ");
            }
        }
    }
    stepOut.close();
    
    delete[] hostTrajectory;
    cudaFree(deviceTrajectory);
    
    
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
void runOptimizationKernel(double lower, double upper, double* hostResults, int* hostIndices, double* hostCoordinates, int N, int MAX_ITER) {
//void runOptimizationKernel(double* hostResults, int N, int dim) {
    printf("first 20 hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }
    printf("\n");
    
    cudaError_t error = launchOptimizeKernel<Function, DIM>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER);
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

    printf("first 20 function values in hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }
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

int main(int argc, char* argv[]) {
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
    const int dim = 2;
    double hostResults[N];// = new double[N];
    std::cout << "number of optimizations = " << N << " max_iter = " << MAX_ITER << " dim = " << dim << std::endl;

    std::cout << std::setprecision(17) << std::scientific;// << std::setprecision(9);
    double f0 = 333777; //sum big  
    for(int i=0; i<N; i++) {
        hostResults[i] = f0;
    }
     
    int indices[N];
    double coordinates[dim];
    for(int i=0; i<N; i++) {
        hostResults[i] = f0;
    }
    std::cout << "\n\n\tRosenbrock Function\n" << std::endl;
    runOptimizationKernel<util::Rosenbrock<dim>, dim>(lower, upper, hostResults, indices, coordinates, N, MAX_ITER); 
    
    
    int hostIndices[N];
    double hostCoordinates[dim];
    for(int i=0; i<N; i++) {
        hostResults[i] = f0;
    }
    std::cout << "\n\n\tRastrigin Function\n"<<std::endl;
    runOptimizationKernel<util::Rastrigin<dim>, dim>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER);
    std::cout << "\n\n\tAckely Function\n"<<std::endl;
    runOptimizationKernel<util::Ackley<dim>, dim>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER);

    std::cout << "\n\n\tGoldStein Price Function\n"<<std::endl;
    runOptimizationKernel<util::GoldsteinPrice<dim>, dim>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER);
    std::cout << "\n\n\tEggholder Function\n"<<std::endl;
    runOptimizationKernel<util::Eggholder<dim>, dim>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER);

    std::cout << "\n\n\tHimmelblau Function\n"<<std::endl;
    runOptimizationKernel<util::Himmelblau<dim>, dim>(lower, upper, hostResults,hostIndices, hostCoordinates, N, MAX_ITER);
    return 0;
}
