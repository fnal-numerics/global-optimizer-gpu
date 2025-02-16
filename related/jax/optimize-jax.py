import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from time import time

def rastrigin(x):
    A = 10
    return A * len(x) + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))

def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# JIT compile the functions
jit_rastrigin = jax.jit(rastrigin)
jit_rosenbrock = jax.jit(rosenbrock)

key = jax.random.PRNGKey(1)
x0_rastrigin = jax.random.uniform(key, shape=(2,), minval=-5.12, maxval=5.12)
#x0_rastrigin = jnp.zeros(2) +0.599613

key_rosenbrock = jax.random.PRNGKey(2)
x0_rosenbrock = jax.random.uniform(key_rosenbrock, shape=(2,), minval=-2.0, maxval=2.0)
#x0_rosenbrock = jnp.zeros(2) + 0.599613

start_time = time()
result_rastrigin = minimize(jit_rastrigin, x0_rastrigin, method='BFGS')
end_time = time()
rastrigin_time = end_time - start_time

start_time = time()
result_rosenbrock = minimize(jit_rosenbrock, x0_rosenbrock, method='BFGS')
end_time = time()
rosenbrock_time = end_time - start_time

print("Rastrigin function")
print(f"Starting point: {x0_rastrigin}")
print(f"Minimum value: {result_rastrigin.fun}")
print(f"Found parameters:")
print("\n".join([f"{xi:.4e}" for xi in result_rastrigin.x]))
print(f"Optimization time: {rastrigin_time:.4f} seconds")
print(f"Number of iterations: {result_rastrigin.nit}")
print(f"Success: {result_rastrigin.success}")
print(f"Number of function evaluations: {result_rastrigin.nfev}")
print(f"Status: {result_rastrigin.status}\n\n")
#print(f"Message: {result_rastrigin.message}\n\n")


print("Rosenbrock function")
print(f"Starting point: {x0_rosenbrock}")
print(f"Minimum value: {result_rosenbrock.fun}")
print("Found parameters:")
print("\n".join([f"{xi:.4e}" for xi in result_rosenbrock.x]))
print(f"Optimization time: {rosenbrock_time:.4f} seconds")
print(f"Number of iterations: {result_rosenbrock.nit}")
print(f"Number of function evaluations: {result_rosenbrock.nfev}")
print(f"Success: {result_rosenbrock.success}")
print(f"Status: {result_rosenbrock.status}")
#print(f"Message: {result_rosenbrock.message}")

