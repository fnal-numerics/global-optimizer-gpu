We thank all reviewers for their constructive feedback. We are encouraged that four reviewers (R1, R3, R4, R5) judged that our paper meets or exceeds the acceptance bar. This rebuttal document aims to answer any questions and is divided into two main sections. We first address common concerns raised across multiple reviewers and then provide detailed reviewer-specific clarifications.  

---

## Common Concerns

### Comparison with Related GPU-based Optimizers (R1–R5)
We acknowledge the concern about missing GPU-based optimizers as the baseline to compare against. Due to space constraints, results comparing **Zeus** with the Julia package *ParallelParticleSwarms (PPS)* were omitted. In our experiments, **Zeus** consistently outperformed PPS, achieving both higher convergence rates and runtime reductions. We will restore these results in the camera-ready version.  

With respect to Barkalov & Gergel's GPU optimizer, their implementation was not publicly available, which prevents us from making a comparison.  

### Novelty of Integration (R2, R5)
The integration of PSO, BFGS, and AD within a GPU framework is non-trivial. Prior work explored only subsets of this pipeline or provided software we were unable to run. **Zeus** requires synchronizing swarm-level explorations with local BFGS optimization steps that integrate automatic differentiation under a single GPU.  

To our knowledge, no prior system combines these three techniques in a unified GPU framework using CUDA C++. While AD itself has been studied for decades, efficient GPU implementations have emerged only recently. We will cite both classical AD work and recent GPU-focused work to position our contribution.  

### Scope of Benchmarks and Practicality (R1, R2, R4)
Although textbook benchmark functions were used to test the performance of **Zeus**, it has also been applied to practical optimization tasks:  
1. Minimizing the negative log-likelihood of a 150-dimensional Gaussian distribution.  
2. Training a small neural network.  

These demonstrate applicability beyond synthetic benchmarks. We will include details of these applications in the revisions.  

### Parallelism Depth and GPU Implementation (R3, R5)
We agree that our current GPU parallelization strategy is straightforward. Importantly, even this approach yields substantial speedups compared to the CPU baseline, validating the practicality of **Zeus**. More sophisticated optimizations—such as parallelized AD and deeper kernel fusion—are natural next steps, which we plan to explore in future versions of **Zeus**.  

---

## Feedback for Individual Reviewers

### Reviewer 1
Thank you for the insightful feedback. Let us respond to each item individually:  

1. **BFGS and discontinuous derivatives:** Thank you for that suggestion. We implemented a template and minimized Bukin N.6 using \(2^{17}\) particles. The global best returned status 0 (surrender) as the gradient norm was well above the set threshold (5.2296772e+08). The coordinates returned were (-9.9999855, 0.9999971), confirming convergence to the minimum.
2. **Finite differences vs AD:** During the CPU version’s implementation, we compared finite differences and automatic differentiation.  Automatic Differentiation provides a more stable option.
3. **Citation:** Thank you for the suggestion; we will cite Bayesian optimization in Section II.  
4. **Practical applications:** We agree to include at least one practical application (log-likelihood minimization or the neural network example).  
5. **Clarifying metrics:** In the current version, \(N_{correct}\) refers to the number of correct solutions that landed in or near the basin of the true minimum across dimensions. A solution was selected if \(Error_{euclidean} < 0.5\). We can modify the box and whiskers to show the best-found function values.  

### Reviewer 2 (most critical)
We would like to thank the reviewer for the detailed feedback. Let us answer each question:  

1. **Why hasn’t PSO+BFGS+AD integration been done before?** Similar approaches do exist, but after extensive communication with the authors we were unable to run their software.  
2. **Previous attempts at partial integrations:** *ParallelParticleSwarms (PPS)* provides PSO+AD+GPU. However, we were not able to use their hybrid PSO+BFGS+AD+GPU method.  
3. **How does Zeus differ?** **Zeus** is implemented in CUDA/C++ and provides a fully working system where all components (PSO+BFGS+AD) are available to users.  
4. **Why no direct comparison?** The closest related algorithm was not working, and the function itself was missing in their repository. We can include our email exchanges with the authors as evidence.  
5. **Next relevant GPU-based method:** We will identify and evaluate alternatives to better contextualize performance.  
6. **Real-world performance:** **Zeus** minimized the negative log-likelihood of a 150-dimensional Gaussian distribution, showcasing a practical higher-dimensional example. A plot will be included.  

### Reviewer 3
We appreciate your feedback. Due to space constraints, we omitted the results against related libraries, which will be added in the camera-ready version.  

For higher-dimensional functions, we plan to explore L-BFGS, which approximates the *N × N* Hessian matrix using *N*-length vectors with past *m* updates, reducing computational complexity from *O(n²)* to *O(mn)*.  

Thank you also for the detailed grammatical and spelling corrections. These will be fixed in the camera-ready version.  

### Reviewer 4
We agree that GPU-to-GPU comparisons are valuable. Unfortunately, Barkalov & Gergel's implementation is not publicly available, but we will note this explicitly.  

As mentioned, we did compare against PPS and will include results. We also agree that distributed-memory extensions are promising and will discuss future directions.  

Practical applications like Gaussian likelihood will also be included.  

### Reviewer 5
We thank the reviewer for recognizing novelty and strong writing. Regarding parallelism, even our straightforward GPU implementation yields substantial speedups. This validates **Zeus** as a practical contribution. We are exploring deeper parallelism as follow-up work. Convergence failure were rare and we are investigating improving the stopping criteria. PPS comparison will also be restored in the camera-ready version.

## Conclusion
In summary, we have clarified the novelty, added evidence of GPU baseline comparisons (PPS), demonstrated practical applications, and resolved technical ambiguities.
With these clarifications, and considering that four reviewers judged the paper to meet or exceed the acceptance bar, we are confident **Zeus** makes a strong contribution to HiPC.
