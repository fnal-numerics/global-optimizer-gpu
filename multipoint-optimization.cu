#include <cstdio>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <fstream>

namespace cuda {

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

__device__ void vector_subtract(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
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

typedef double (*rosen_fun)(double* x, int n);
typedef void (*rosen_der)(double* x, double* g, int n);

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


__device__ void initialize_identity_matrix_device(double* H, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

__device__ void bfgs_device(double* H, double* delta_x, double* delta_g, double delta_dot, int n) {
    const int dim = 2;
    if (fabs(delta_dot) < 1e-10) return;  // protect against division by zero
    
    double yt_s = 1.0 / delta_dot; // y^T 
    if (fabs(yt_s) < 1e-10 || isnan(delta_dot) || isinf(delta_dot)) {
        //printf("yt_s is too close to zero, resseting H to identity matrix: %f", yt_s);
	initialize_identity_matrix_device(H, n);
        return;
    }

    double term1[dim * dim]; // s y^T
    double term2[dim * dim]; // y s^T
    double term3[dim * dim]; // s s^T

    outer_product_device(delta_x, delta_g, term1, n);
    outer_product_device(delta_g, delta_x, term2, n);
    outer_product_device(delta_x, delta_x, term3, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int index = i * n + j;
            double I_ij = (i == j) ? 1.0 : 0.0;
            H[index] = (I_ij - term1[index] * yt_s) * H[index] * (I_ij - term2[index] * yt_s) + term3[index] * yt_s;
        }
    }
}


/* Support Function for Line Search
 * 	directional derivative
 *	put_in_range
 *	poly_min_extrap
 **/

__device__ double directional_derivative(const double *grad, const double *p, int dim) {
    double d = 0.0;
    for (int i = 0; i < dim; ++i) {
        d += grad[i] * p[i];
    }
    return d;
}


template <typename T>
__device__ T put_in_range(const T& a, const T& b, const T& val) {
    if (a < b) {
        if (val < a) return a;
        else if (val > b) return b;
    } else {
        if (val < b) return b;
        else if (val > a) return a;
    }
    return val;
}

__device__ double put_in_range_double(const double& a, const double& b, const double& val) {
    return put_in_range<double>(a, b, val);
}

/*  Polynomial Minimum Extrapolation
 *
 *  This function uses a cubic polynomial interpolation based on two points and 
 *  their derivatives to estimate the minimum of the polynomial within a given interval.
 *  used when we have function values and gradients at the initial point and another point
 *  along the search direction.
 *  The function estimates the point x where the polynomial potentially reaches a minimum
 */
__device__ double poly_min_extrap(double f0, double d0, double f1, double d1, double limit = 1) {
    double n = 3 * (f1 - f0) - 2 * d0 - d1;
    double e = d0 + d1 - 2 * (f1 - f0);

    double temp = fmax(n * n - 3 * e * d0, 0.0);

    if (temp < 0) return 0.5;

    temp = sqrt(temp);

    if (fabs(e) <= 1e-15) return 0.5;

    double x1 = (temp - n) / (3 * e);
    double x2 = -(temp + n) / (3 * e);

    double y1 = f0 + d0 * x1 + n * x1 * x1 + e * x1 * x1 * x1;
    double y2 = f0 + d0 * x2 + n * x2 * x2 + e * x2 * x2 * x2;

    double x = (y1 < y2) ? x1 : x2;

    return put_in_range_double(0.0, limit, x);
}

// 3 argument version
/*
 * this function uses a quadratic interpolation based on the function value and 
 * derivative at the start and the function value at another point.
 */ 
__device__ double poly_min_extrap(double f0, double d0, double f1) {
    double temp = 2 * (f1 - f0 - d0);
    if (fabs(temp) <= d0 * 1e-15) return 0.5;

    double alpha = -d0 / temp;
    return put_in_range_double(0.0, 1.0, alpha);
}


/* 6 argument version
 * 
 * this function constructs a cubic polynomial based on function values
 * at three points without requiring their derivatives.
 * by considering more points -> better fit and more reliable estimate for
 * the minimum, especially in non-smooth or complex functions
 */ 
__device__ double poly_min_extrap(double f0, double d0, double x1, double f_x1, double x2, double f_x2) {
    const double aa2 = x2 * x2;
    const double aa1 = x1 * x1;
    double temp = aa2 * aa1 * (x1 - x2);

    if (temp == 0 || isinf(1.0 / temp)) return x1 / 2.0;  // Check for subnormal or zero division

    double m11 = aa2, m12 = -aa1;
    double m21 = -aa2 * x2, m22 = aa1 * x1;
    double v1 = f_x1 - f0 - d0 * x1;
    double v2 = f_x2 - f0 - d0 * x2;

    double a = (m11 * v1 + m12 * v2) / temp;
    double b = (m21 * v1 + m22 * v2) / temp;

    temp = b * b - 3 * a * d0;

    if (temp < 0 || a == 0) return (f0 < f_x2) ? 0 : x2;

    double solution = (-b + sqrt(fmax(temp, 0.0))) / (3 * a);
    return put_in_range(0.0, x2, solution);
}


__device__ double dlib_line_search(rosen_fun fun, rosen_der der, double f0, double d0, double rho, double sigma, double min_f, int max_iter, double *x, double *p, double lambda, int dim)
{
    if (!(0 < rho && rho < sigma && sigma < 1 && max_iter > 0)) {
        printf("Invalid arguments provided to line_search\n");
        return -1;
    }

    // 1 <= tau1a < tau1b controls the alpha jump size during the bracketing phase of the search
    const double tau1a = 1.4;
    const double tau1b = 9;
    // The bracketing phase of this function is implemented according to block 2.6.2 from
    // the book Practical Methods of Optimization by R. Fletcher. The sectioning 
    // phase is an implementation of 2.6.4 from the same book.

    // it must be the case that 0 < tau2 < tau3 <= 1/2 for the algorithm to function
    // correctly but the specific values of tau2 and tau3 aren't super important.
    const double tau2 = 1.0/10.0;
    const double tau3 = 1.0/2.0;

    if (fabs(d0) <= fabs(f0) * 1e-10) // epsilon for double precision
        return 0;

    if (f0 <= min_f)
        return 0;

    const double mu = (min_f - f0) / (rho * d0);

    double alpha = 1;
    if (mu < 0)
        alpha = -alpha;
    alpha = min(max(0.0, alpha), 0.65 * mu);

    double last_alpha = 0;
    double last_val = f0;
    double last_val_der = d0;

    double a, b, a_val, b_val, a_val_der, b_val_der;
    const double thresh = fabs(sigma * d0);

    unsigned long itr = 0;
    double x_alpha[2];
    double g[2];
    
    // do the bracketing stage to find the bracket range [a,b]
    while (true) {
        ++itr;
        for (int i = 0; i < dim; ++i) {
            x_alpha[i] = x[i] + alpha * p[i];
        }
        const double val = fun(x_alpha, dim);
        der(x_alpha, g, dim);

        const double val_der = directional_derivative(g, p, dim);
        
	// we are done with the line search since we found a value smaller
	// than the minimum f value
	if (val <= min_f) {
            return alpha;
        }

        if (val > f0 + rho * alpha * d0 || val >= last_val) {
            a_val = last_val;
            a_val_der = last_val_der;
            b_val = val;
            b_val_der = val_der;

            a = last_alpha;
            b = alpha;
            break;
        }

        if (fabs(val_der) <= thresh) {
            return alpha;
        }
	// if we are stuck not making progress then quit with the current alpha
        if (last_alpha == alpha || itr >= max_iter) {
            return alpha;
        }
        if (val_der >= 0) {
            a_val = val;
            a_val_der = val_der;
            b_val = last_val;
            b_val_der = last_val_der;

            a = alpha;
            b = last_alpha;
            break;
        }

        const double temp = alpha;
        double first, last;
        if (mu > 0) {
            first = fmin(mu, alpha + tau1a * (alpha - last_alpha));
            last = fmin(mu, alpha + tau1b * (alpha - last_alpha));
        } else {
            first = fmax(mu, alpha + tau1a * (alpha - last_alpha));
            last = fmax(mu, alpha + tau1b * (alpha - last_alpha));
        }

	// pick a point between first and last by doing some kind of interpolation
	if (last_alpha < alpha) 
	    alpha = last_alpha + (alpha-last_alpha)*poly_min_extrap(last_val, last_val_der, val, val_der, 1e10);
        else
	    alpha = alpha + (last_alpha-alpha)*poly_min_extrap(val, val_der, last_val, last_val_der, 1e10);
	alpha = put_in_range(first, last, alpha);

        last_alpha = temp;
        last_val = val;
        last_val_der = val_der;
    }//end bracketing while
    
    // now do the sectioning phase from 2.6.4
    while(true) {
        ++itr;
        double first = a + tau2*(b-a);
        double last = b - tau3*(b-a);

        // use interpolation to pick alpha between first and last
        alpha = a + (b-a)*poly_min_extrap(a_val, a_val_der, b_val, b_val_der);
        alpha = put_in_range(first,last,alpha);

        for (int i = 0; i < dim; ++i) {
            x_alpha[i] = x[i] + alpha * p[i];
        }
        const double val = fun(x_alpha, dim);
        der(x_alpha, g, dim);
	const double val_der = directional_derivative(g,p,dim);

        // we are done with the line search since we found a value smaller
        // than the minimum f value or we ran out of iterations.
        if (val <= min_f || itr >= max_iter) {
            return alpha;
        }
        // stop if the interval gets so small that it isn't shrinking any more due to rounding error 
        if (a == first || b == last)
        {
            return b;
        }

        // If alpha has basically become zero then just stop.  Think of it like this,
        // if we take the largest possible alpha step will the objective function
        // change at all?  If not then there isn't any point looking for a better
        // alpha.
        const double max_possible_alpha = fmax(fabs(a),fabs(b));
        if (fabs(max_possible_alpha*d0) <= fabs(f0)*1e-10) {
	    return alpha;
        }

        if (val > f0 + rho*alpha*d0 || val >= a_val)
        {
            b = alpha;
            b_val = val;
            b_val_der = val_der;
        }
        else
        {
            if (fabs(val_der) <= thresh)
	        return alpha;

            if ( (b-a)*val_der >= 0)
            {
                b = a;
                b_val = a_val;
                b_val_der = a_val_der;
            }

            a = alpha;
            a_val = val;
            a_val_der = val_der;
        }//end else
    }// end secioning while

    printf("we down the bottom rookie\n");
    // Cleanup
    return alpha;
}// end dlib_line_search

__global__ void optimizeKernel(double* devicePoints,double* deviceResults, int N,int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int early_stopping = 0;
    double H[2 * 2];
    double g[2], x[2], x_new[2], p[2], g_new[2], delta_x[2], delta_g[2], new_direction[2];
    for (int i = 0; i < 4; i++) H[i] = 0;
    for (int i = 0; i < 2; i++) {
        g[i] = 0;
        x[i] = 0;
        x_new[i] = 0;
        p[i] = 0;
        g_new[i] = 0;
        delta_x[i] = 0;
        delta_g[i] = 0;
    }
    double minima = 10000000;
    double min_alpha = 1e-4;
    double max_alpha = 1.0;
    double delta_dot = 0;
    // Line Search params
    double min_f = 1e-12;
    int max_iter = 100;
    double lambda = 15.0;
    double alpha = 1.0;
    double d0 = 0;

    initialize_identity_matrix(H, dim);
    //printf("past identity");

    for (int i = 0; i < dim; ++i) {
        x[i] = devicePoints[i * dim + idx];
        //printf("x[%d] = ",idx);
	//printf(" %f ", x[i]);
    }
    
    double f0 = rosenbrock_device(x, dim);
    if (idx == 0) {
       printf("\n\nf0 = %f", f0);
    }

    
    for (int iter = 0; iter < 25; ++iter) {
	//printf("idx = %d", idx);
	//printf("x[0] = %f", x[0]);
	//printf("\nit#%d\n",iter);
	rosenbrock_gradient_device(x, g, dim);
        d0 = 0.0;
        // compute the search direction p = -H * g
        for (int i = 0; i < dim; ++i) {
            //printf("\ng[%d] = %f ",i,g[i]);
       	    p[i] = 0.0;
            for (int j = 0; j < dim; ++j) {
                p[i] -= H[i * dim + j] * g[j]; // i * dim + j since H is flattened arr[]
            }    
	    d0 += g[i] * p[i]; // get current directional derivatives d0 while we at it
        }

        // determining the optimal step size is crucial in an optimization routine
        // alpha = 0.5 based on taylor series expansion
	//alpha = line_search(x, p, g, dim, 0.001, 0.9, 0.0011); // alpha_init ,rho, c
	// note that tweaking rho and sigma cna significantly impact the search behavior
	//   these variable control the aggressiveness and criteria for accepting proper step size
	// for example, smaller rho makes armijo condition easier to satisfy potentially accepting
	// suboptimal alpha. on the contrary, larger rho demands a more significant decrease in the
	// objective function
	//  sigma controls curvature that makes sure that the slope at new point ha simproved sufficiently from last point. 
	// small sigma = accept alpha that don't significantly change the gradient
	// large sigma = need greater improvement in gradient which means more stable convergence
        // potentially making the search slower
	alpha = dlib_line_search(rosenbrock_device, rosenbrock_gradient_device, f0, d0, 0.17, 0.87, min_f, max_iter, x, p, lambda, dim);
    // Use the alpha obtained from the line search
	//printf("new step size = %f", alpha);
	if(!valid(alpha)) {
	    //printf("alpha not valid, reset to 1.0");
	    alpha = 1.0;
	}
	if(alpha < min_alpha)
	    alpha = min_alpha;
	else if (alpha > max_alpha)
     	    alpha = max_alpha;
	
	// update current point x by taking a step size of alpha in the direction p
	for (int i = 0; i < dim; ++i) {
            x_new[i] = x[i] + alpha * p[i];
            //printf("new x val = %f ", x_new[i]);
	}
 
    // get the new gradient g_new at x_new
    rosenbrock_gradient_device(x_new, g_new, dim);

    // calculate new delta_x and delta_g
    for (int i = 0; i < dim; ++i) {
        delta_x[i] = x_new[i] - x[i]; // differentce between he new point and old point
        delta_g[i] = g_new[i] - g[i]; // difference in gradient at the new point vs old point
    }

    // bfgs update on H
    delta_dot = dot_product_device(delta_x, delta_g, dim);
    if (fabs(delta_dot) <= 1e-7) {
    if (alpha == min_alpha)
		alpha *= 2.5;
    else
		alpha = 0.5;
    //printf("alpha changed to %f ", alpha);
    vector_scale(p, alpha, new_direction, dim);
    vector_add(new_direction, x, x_new, dim);
    if (early_stopping < 10)
        early_stopping++;
    else {
		//printf("\nearly stopping trigered");
		break;
    }
	}

	bfgs_device(H, delta_x, delta_g, delta_dot, dim);

        // only update x and g for next iteration if the new minima is smaller than previous
	double min = rosenbrock_device(x_new, dim);
        if (min < minima) {
	    for(int i=0; i<dim; ++i)
		x[i] = x_new[i];
	    minima = min;
	}
    }// end outer for
    minima = rosenbrock_device(x, dim);
    //printf("\nmax iterations reached, predicted minima = %f\n", minima);
    deviceResults[idx] = minima;
}// end optimizerKernel


} // end of namespace cuda


bool readDoublesFromFile(const char* filename, double* data, int N) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Failed to open file for reading.\n";
        return false;
    }

    inFile.read(reinterpret_cast<char*>(data), N * sizeof(double));
    if (inFile.gcount() != N * sizeof(double)) {
        std::cerr << "Failed to read the expected number of doubles.\n";
        inFile.close();
        return false;
    }

    inFile.close();
    return true;
}

cudaError_t launchOptimizeKernel(double* hostResults, int N, int dim) {
    dim3 blockSize(128); // Use 128 threads per block as Dr. Zubair told me
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x); // make sure all instances are covered
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per grid: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    double* deviceResults;
    double* devicePoints;// = generate_random_numbers(N);
    
    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double* hostPoints = new double[N * dim];
    const char* filename = "randomNumbers.bin";
    
    if (!readDoublesFromFile(filename, hostPoints, N*dim)) {
 	delete[] hostPoints;
        return cudaErrorUnknown;
    } else {
	printf("Data file read with %d points\n",N*dim);
    }

    cudaError_t allocStatus = cudaMalloc(&devicePoints, N * dim * sizeof(double));
    if (allocStatus != cudaSuccess) return allocStatus;
    allocStatus = cudaMalloc(&deviceResults, N * sizeof(double));
    if (allocStatus != cudaSuccess) {
        cudaFree(devicePoints); // deallocate memory if error
        return allocStatus;
    }
    // copy host points to device points
    cudaMemcpy(devicePoints, hostPoints, N * dim * sizeof(double), cudaMemcpyHostToDevice);
    // launch the actual kernel in cuda namespace

    // timer tightly wrapped around kernel launch
    cudaEventRecord(start);
    cuda::optimizeKernel<<<numBlocks, blockSize>>>(devicePoints, deviceResults, N, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliKernel = 0;
    cudaEventElapsedTime(&milliKernel, start, stop);
    printf("\nOptimization Kernel execution time = %f ms\n", milliKernel);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        cudaFree(deviceResults);
        cudaFree(devicePoints);        
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(deviceResults);
        cudaFree(devicePoints);
	return err;
    }
    
    // check for errors after launch and synchronize
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(deviceResults);
        cudaFree(devicePoints);
        return error;
    }

    // copy results back to host
    error = cudaMemcpy(hostResults, deviceResults, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(deviceResults);
        cudaFree(devicePoints);
        return error;
    }
    printf("results are copied back to host\n");

    // deallocate device memory
    cudaFree(deviceResults);
    cudaFree(devicePoints);
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
            double tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    double tmp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = tmp;
    return i + 1;
}

void quickSort(double arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
void runOptimizationKernel(double* hostResults, int N, int dim) {
    printf("first 20 hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }
    printf("\n");
    
    cudaError_t error = launchOptimizeKernel(hostResults, N, dim);
    if (error != cudaSuccess) {
        printf("CUDA error: %s", cudaGetErrorString(error));
    } else {
        printf("\nSuccess!! No Error!\n");
    }

    //quickSort(hostResults, 0, N - 1);
    //printf("after sorting the array: \n");    

    printf("first 20 function values in hostResults\n");
    for(int i=0;i<20;i++) {
       printf(" %f ",hostResults[i]);
    }
    printf("\n");

}
double square(double x){ return x*x;}

double rosenbrock(double* x, int dim) {
    double sum = 0.0;
    for(size_t i=0;i<dim;++i) {
        sum += 100 * square(x[i+1] - square(x[i])) + square(1 - x[i]);
    }
    return sum;
}

int main() {
    //const char* filename = "randomNumbers.bin";

    const int N = 1024*128;
    const int dim = 2;
    double hostResults[N];
    std::cout << "number of optimizations = " << N << std::endl;

    double x0[2] = {5.0, 5.0};
    double f0 = rosenbrock(x0, dim);    
    for(int i=0; i<N; i++) {
        hostResults[i] = f0;
    }

    runOptimizationKernel(hostResults, N, dim);

    return 0;
}

