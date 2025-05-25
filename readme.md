# CUDA Rayleigh-Taylor Instability Simulation

A high-performance numerical simulation of the Rayleigh-Taylor instability implemented in CUDA, Python, and C for comparative performance analysis.

## Project Overview

This project implements a numerical simulation of the Rayleigh-Taylor instability using CUDA parallel computing. The Rayleigh-Taylor instability occurs when a heavy fluid rests on top of a lighter fluid under the influence of gravity, creating an unstable interface that evolves over time. The simulation solves the compressible fluid dynamics equations on a 2D grid, parallelizing computations on GPU for optimal performance.

## Simulation Results

<div align="center">
  <img src="rayleigh_taylor_instability_512x1024.gif" width="300" alt="Rayleigh-Taylor Instability Evolution">
</div>

*Density field evolution showing the characteristic mushroom-like structures of the Rayleigh-Taylor instability (512×1024 grid resolution)*

## Repository Contents

This repository contains three main implementation approaches:

- **`rayleigh_taylor_instability_cpu_py.ipynb`** - Pure Python implementation with NumPy for baseline comparison
- **`rayleigh_taylor_instability_cpu_c.ipynb`** - C CPU implementation for performance comparison
- **`rayleigh_taylor_instability_gpu.ipynb`** - CUDA GPU-accelerated implementation for maximum performance

## Mathematical Framework

### Conservative Variables

The system models a compressible fluid with the following conservative variables:
- **Density**: ρ (denoted as r)
- **Momentum**: ρu (denoted as ru) and ρv (denoted as rv)  
- **Total Energy**: e (denoted as e)

### Governing Equations

The 2D conservation equations are:

#### Continuity (Mass Conservation)
$$\frac{\partial \rho}{\partial t} + \frac{\partial(\rho u)}{\partial x} + \frac{\partial(\rho v)}{\partial y} = k_1\left(\frac{\partial^2 \rho}{\partial x^2} + \frac{\partial^2 \rho}{\partial y^2}\right)$$

where $k_1$ is an artificial diffusion coefficient.

#### Momentum Conservation (x-direction)
$$\frac{\partial(\rho u)}{\partial t} + \frac{\partial(\rho u u)}{\partial x} + \frac{\partial(\rho u v)}{\partial y} + \frac{\partial p}{\partial x} = k_2\frac{\partial^2(\rho u)}{\partial x^2}$$

where $p$ is pressure and $k_2$ is a diffusion coefficient.

#### Momentum Conservation (y-direction)
$$\frac{\partial(\rho v)}{\partial t} + \frac{\partial(\rho u v)}{\partial x} + \frac{\partial(\rho v v)}{\partial y} + \frac{\partial p}{\partial y} = -g_a\rho + k_2\frac{\partial^2(\rho v)}{\partial y^2}$$

where $g_a = -10$ is the gravitational acceleration.

#### Total Energy Conservation
$$\frac{\partial e}{\partial t} + \frac{\partial(u(e + p))}{\partial x} + \frac{\partial(v(e + p))}{\partial y} = -g_a\rho v + k_3\left(\frac{\partial^2 e}{\partial x^2} + \frac{\partial^2 e}{\partial y^2}\right)$$

where $e = \frac{p}{\gamma-1} + \frac{1}{2}\rho(u^2 + v^2)$ is the total energy (internal + kinetic), with $\gamma = 1.4$ (specific heat ratio), and $k_3$ is a diffusion coefficient.

### Initial and Boundary Conditions

#### Initial Conditions
The initial interface is defined at $y = L_y/2$:
- **Density**: $\rho = 2.0$ for $y \geq L_y/2$, $\rho = 1.0$ for $y < L_y/2$
- **Velocity**: $u = 0$ everywhere
- **Velocity**: $v = 0$ except near interface ($|y - L_y/2| \leq 0.05$), where $v$ is a random perturbation between $-10^{-3}$ and $10^{-3}$
- **Pressure**: $p = 40 + \rho g_a(y - L_y/2)$ (hydrostatic equilibrium)
- **Energy**: $e$ derived from $p$, $u$, and $v$

#### Boundary Conditions
- **Y-boundaries** (top and bottom): Rigid walls (normal velocity $v = 0$, hydrostatic equilibrium for $p$)
- **X-boundaries**: Periodic (values at $x = 0$ and $x = L_x$ are connected)

### Time Integration Methods

Two numerical schemes are implemented:

#### 1. Explicit Euler
$$q^{n+1} = q^n + \Delta t \cdot \text{RHS}(q^n)$$

#### 2. Second-Order Runge-Kutta (RK2)
$$q^* = q^n + \Delta t \cdot \text{RHS}(q^n)$$
$$q^{n+1} = \frac{1}{2}q^n + \frac{1}{2}(q^* + \Delta t \cdot \text{RHS}(q^*))$$

where $q = [\rho, \rho u, \rho v, e]$ and RHS represents the right-hand side of the equations.

#### CFL Condition
The time step $\Delta t$ must satisfy the CFL condition:
$$\Delta t = C \frac{\Delta x}{\max(|u|, |v|, c)}$$
where $c = \sqrt{\frac{\gamma p}{\rho}}$ is the speed of sound and $C = 0.2$.

### Artificial Diffusion
For numerical stability, diffusion terms are included:
$$k_1 = 0.0125 \cdot \frac{\Delta x^2}{2\Delta t}, \quad k_2 = 0.125 \cdot \frac{\Delta x^2}{2\Delta t}, \quad k_3 = 0.0125 \cdot \frac{\Delta x^2}{2\Delta t}$$

## Performance Analysis

### Implementation Comparison

| Implementation | Relative Performance | Memory Usage | Parallelization |
|----------------|---------------------|--------------|-----------------|
| **Python/NumPy** | 1x (baseline) | High | Limited (vectorized ops) |
| **C CPU** | ~10-50x faster | Moderate | Sequential loops |
| **CUDA GPU** | ~100-500x faster | Optimized | Thousands of threads |

### Key Performance Insights

**Python Implementation:**
- Easy to prototype and debug
- Limited by Python's GIL and interpreted nature
- Suitable for algorithm validation and small grids

**C CPU Implementation:**
- Significant performance improvement over Python
- Efficient memory management with direct array access
- Sequential implementation without threading
- Tested on smaller grids (64×128) for faster development

**CUDA GPU Implementation:**
- Massive parallel acceleration with GPU kernels
- Shared memory optimization for reduction operations
- Block-thread organization for spatial computations
- Optimal for production simulations on large grids (512×1024)

## Grid Size Analysis

Performance scaling with different grid resolutions:

| Grid Size | Memory Required | CUDA Performance | Recommended Use Case |
|-----------|----------------|------------------|---------------------|
| 256×128 | ~50 MB | Excellent | Development/Testing |
| 512×256 | ~200 MB | Excellent | Standard simulations |
| 1024×512 | ~800 MB | Very Good | High-resolution studies |
| 2048×1024 | ~3.2 GB | Good* | Research applications |

*Performance depends on GPU memory bandwidth and compute capability.

## Numerical Method Comparison

### Euler vs RK2 Analysis

**Explicit Euler Method:**
- Simple implementation
- Low computational cost per step
- First-order accuracy
- More restrictive stability conditions

**Second-Order Runge-Kutta (RK2):**
- Higher accuracy (second-order)
- Better stability properties
- More physical results
- Double computational cost per step
- More complex implementation

**Recommendation:** RK2 provides significantly better accuracy and stability, making it the preferred choice despite the computational overhead.

## Key Features

- **GPU Acceleration**: Up to 500x speedup over Python implementation
- **Multiple Methods**: Both Euler and RK2 time integration schemes
- **Comprehensive Benchmarking**: Performance comparison across Python, C, and CUDA
- **CUDA Optimizations**: Parallel kernel execution and shared memory for reductions
- **Configurable Parameters**: Adjustable grid sizes, diffusion coefficients, and time stepping
- **Scalable Design**: Efficient performance from small test cases to large production runs

## Benchmarking Environment

All performance benchmarks were conducted on:
- **Platform**: Kaggle Notebooks
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **CPU**: Intel Xeon (4 cores)
- **Memory**: 13GB RAM

## Limitations and Future Work

### Current Limitations
- **C Implementation**: Sequential processing without OpenMP parallelization
- **CUDA Optimization**: Shared memory currently limited to reduction operations
- **Memory Management**: Dynamic allocation patterns could be further optimized

### Future Improvements
- **Enhanced Parallelization**: Add OpenMP threading to C implementation
- **CUDA Optimizations**: Implement shared memory for spatial derivative computations
- **Memory Efficiency**: Explore texture memory for read-only data access
- **Adaptive Methods**: Implement adaptive time stepping and mesh refinement
- **Validation Studies**: Compare results against analytical solutions and experimental data

## Contact & Support

For detailed performance reports, technical questions, or collaboration opportunities, please contact me.

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

## License

This project is available under the MIT License. See LICENSE file for details.

## References

- Rayleigh-Taylor Instability Theory
- CUDA Programming Guide

---

*Developed as part of advanced parallel programming coursework focusing on GPU-accelerated numerical simulations.*