We thank all reviewers for their constructive feedback. We are encouraged that four reviewers (R1, R3, R4, R5) judged that our paper meets or exceeds the acceptance bar. This rebuttal document aims to provide clarifications and answer any questions. It is divided into two main sections. We first address common concerns raised across multiple reviewers and then provide detailed reviewer-specific clarifications. 

---

## Common Concerns

### Comparison with Related GPU-based Optimizers (R1–R5)
We acknowledge the concern about missing GPU-based optimizers as the baseline to compare against. 
Due to space constraints, results comparing **Zeus** with the Julia package *ParallelParticleSwarms (PPS)*} were omitted. 
After extensive communications with the authors of that work, we were unable to run their main hybrid.
For this reason, we were only able to compare against two ingredients of our method.
In their code, they also mention race condition in their implementation, which may produce spurious results. 
Their solution to remove the race condition made it no longer a true particle swarm algorithm. 
In our experiments, **Zeus** consistently outperformed PPS, achieving both higher convergence rates and runtime reductions.
We will restore these results in the camera-ready version. 
With respect to Barkalov & Gergel's GPU optimizer, their implementation was not publicly available, which prevents us to compare.
Other prior work either lack one or two core ingredient of our method or dooes not provide an open-source implementations. 
This absence of an open-source, fully functional CUDA/C++ integration of PSO, BFGS, AD, and GPU acceleration is precisely the gap **Zeus** fills. 

### Parallelism Depth (R2, R3, R5)
We agree that our current GPU parallelization strategy is relatively straightforward.
For this reason, we will implement parallel AD to the final paper.
The code already exist for parallelizing AD across dimensions, it just has not been integrated into **Zeus**. 
We will include results comparing sequential and parallel AD in the camera-ready version.

### Scope of Benchmarks and Practicality (R1, R2, R4)
Even though textbook benchmark functions were used to test the performance of **Zeus**, it can be applied to practical optimizations tasks like minimizing the negative log-likelihood of a 150-dimensional Gaussian distribution.
Even though high-dimensional Gaussian fitting is a real-world problem, it might be seen as too simple.
In high-energy physics (HEP), researchers frequently use minimum negative log-likelihood with Poisson distributions to fit parameters to histogrammed data?, where the theory predicts probabilities of observations as a function of energy??
We are currently working on this as we need to simulate data???
We will include the details of these applications in the revisions.

### Novelty of Integration (R2, R5)
The integration of PSO, BFGS, and AD within a GPU framework is non-trivial. Prior work explored only subsets of this pipeline or provided software we were unable to run. **Zeus** requires synchronizing swarm-level explorations with local BFGS optimization steps that integrate automatic differentiation under a single GPU.  

To our knowledge, no prior system combines these three techniques in a unified GPU framework using CUDA C++. While AD itself has been studied for decades, efficient GPU implementations have emerged only recently. We will cite both classical AD work and recent GPU-focused work to position our contribution.  

---

## Feedback for Individual Reviewers

### Reviewer 1
Thank you for the insightful feedback. Let us respond to each item individually:  

1. BFGS does suffer from functions with discontinuous derivatives. Thank you for that suggestion. We implemented a template for Bukin N.6 and ran using \(2^{17}\) particles. Since the global minimum is located at a discontinuity, **Zeus** returns the "surrender" status, meaning that we have reached the maximum number of iterations as the norm of the gradient near the minimum (5.2296772e+08) is well above the set threshold. Regardless, the coordinates returned were (-9.9999855e+00 9.9999710e-01) with function value (9.89498e-06), confirming convergence to the global minimum.
2. During the CPU version’s implementation, we compared finite differences and automatic differentiation. Due to space constraint, and automatic differentiation provides a more stable option, it was omitted.
3.Thank you for the suggestion; we will cite Bayesian optimization paper in Section II.  
4. We agree to include at least one practical application in addition to synthetic functions. We will include in the revision the log-likelihood minimization.
5. In the current version, \(N_{correct}\) refers to the number of correct solutions that landed in or near the basin of the true minimum across dimensions. A solution was selected if \(Error_{euclidean} < 0.5\). We can modify the box and whiskers to show the best-found function values.  

### Reviewer 2 (most critical)
We would like to thank the reviewer for the detailed and thoughtful feedback. Below is our answer for each question:  

1. **Why hasn’t PSO+BFGS+AD integration been done before?**
Several approaches have explored a subset of these (e.g., PSO+GPUs, BFGS with AD), but to the best of our knowledge no prior implementation provides all three in a single GPU-framework that works.
2. **Prior combinations of techniques**
Combinations like PSO+BFGS (e.g., the hybrid *ParallelParticleSwarms (PPS)* method) have been proposed. PSO+AD is less meaningful, as PSO is gradient-free. *PPS* in principle combines PSO+BFGS+AD+GPU, but in practice the available implementation was not functional. We reached out to the authors, but unfortunately did not receive helpful response. 
3. **How Zeus differ?**
**Zeus** is a CUDA/C++ implementation that integrates PSO and BFGS with forward-mode AD in a fully functional GPU framework. To our knowledge, it is the first system that allows practitioners to leverage all these components from a C++ program.
4. **Why no direct comparison?**
 The PPS algorithm was most related work, but the available code was not fully functional, and even contained unresolved issues like race conditions. We have communicated with the authors several times, but were unable to get a helpful response. For this reason, and the space constraint, we chose to focus on comparison against CPU version. We will make this limitation clearer in the final version.
5. **Evaluating against other GPU-based methods**
We agree that broader context is valuable. We will include comparisons against the Julia-based package.
6. **Real-world performance**
Beyond selected benchmarks, **Zeus** successfully minimized the negative log-likelihood of a 150-dimensional Gaussian distribution. It may be seen as simplistic real-world, so we are including a real-world HEP problem we described in the first section. We will add these results into the final manuscript.

### Reviewer 3
We appreciate your feedback. Due to space constraints, we omitted the results against related libraries, which will be added in the camera-ready version.  

For higher-dimensional functions, we plan to explore L-BFGS, which approximates the *N × N* Hessian matrix using *N*-length vectors with past *m* updates, reducing computational complexity from *O(n²)* to *O(mn)*.  

Thank you also for the detailed grammatical and spelling corrections. These will be fixed in the camera-ready version.  

### Reviewer 4
We agree that GPU-to-GPU comparisons are valuable. Unfortunately, Barkalov & Gergel's implementation is not publicly available, but we will note this explicitly.  

As mentioned, we did compare against *PPS* and will include results. We also agree that distributed-memory extensions are promising and will discuss future directions.  

A real-world HEP application will also be included. 

### Reviewer 5
We thank the reviewer for recognizing novelty and strong writing. Regarding parallelism, even our straightforward GPU implementation yields substantial speedups. This validates **Zeus** as a practical contribution. We are exploring deeper parallelism to be included in the final version. We have an implementation 

Convergence failure were rare and we are investigating improving the stopping criteria. *PPS* comparison will also be restored in the camera-ready version.

## Conclusion
In summary, we have clarified the novelty, added evidence of GPU baseline comparisons (PPS), demonstrated practical applications, and resolved technical ambiguities.
With these clarifications, and considering that four reviewers judged the paper to meet or exceed the acceptance bar, we are confident **Zeus** makes a strong contribution to HiPC.
