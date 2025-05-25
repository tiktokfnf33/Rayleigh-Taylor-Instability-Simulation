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
- **`rayleigh_taylor_instability_cpu_c.ipynb`** - Optimized C CPU implementation for performance comparison
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
```
∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y = k₁(∂²ρ/∂x² + ∂²ρ/∂y²)
```
where k₁ is an artificial diffusion coefficient.

#### Momentum Conservation (x-direction)
```
∂(ρu)/∂t + ∂(ρuu)/∂x + ∂(ρuv)/∂y + ∂p/∂x = k₂∂²(ρu)/∂x²
```
where p is pressure and k₂ is a diffusion coefficient.

#### Momentum Conservation (y-direction)
```
∂(ρv)/∂t + ∂(ρuv)/∂x + ∂(ρvv)/∂y + ∂p/∂y = -gₐρ + k₂∂²(ρv)/∂y²
```
where gₐ = -10 is the gravitational acceleration.

#### Total Energy Conservation
```
∂e/∂t + ∂(u(e + p))/∂x + ∂(v(e + p))/∂y = -gₐρv + k₃(∂²e/∂x² + ∂²e/∂y²)
```
where e = p/(γ-1) + ½ρ(u² + v²) is the total energy (internal + kinetic), with γ = 1.4 (specific heat ratio), and k₃ is a diffusion coefficient.

### Initial and Boundary Conditions

#### Initial Conditions
The initial interface is defined at y = Ly/2:
- **Density**: ρ = 2.0 for y ≥ Ly/2, ρ = 1.0 for y < Ly/2
- **Velocity**: u = 0 everywhere
- **Velocity**: v = 0 except near interface (|y - Ly/2| ≤ 0.05), where v is a random perturbation between -10⁻³ and 10⁻³
- **Pressure**: p = 40 + ρgₐ(y - Ly/2) (hydrostatic equilibrium)
- **Energy**: e derived from p, u, and v

#### Boundary Conditions
- **Y-boundaries** (top and bottom): Rigid walls (normal velocity v = 0, hydrostatic equilibrium for p)
- **X-boundaries**: Periodic (values at x = 0 and x = Lx are connected)

### Time Integration Methods

Two numerical schemes are implemented:

#### 1. Explicit Euler
```
q^(n+1) = q^n + Δt · RHS(q^n)
```

#### 2. Second-Order Runge-Kutta (RK2)
```
q* = q^n + Δt · RHS(q^n)
q^(n+1) = ½q^n + ½(q* + Δt · RHS(q*))
```

where q = [ρ, ρu, ρv, e] and RHS represents the right-hand side of the equations.

#### CFL Condition
The time step Δt must satisfy the CFL condition:
```
Δt = C · Δx/max(|u|, |v|, c)
```
where c = √(γp/ρ) is the speed of sound and C = 0.2.

### Artificial Diffusion
For numerical stability, diffusion terms are included:
```
k₁ = 0.0125 · Δx²/(2Δt)
k₂ = 0.125 · Δx²/(2Δt)  
k₃ = 0.0125 · Δx²/(2Δt)
```

## Performance Analysis

### Implementation Comparison

| Implementation | Relative Performance | Memory Usage | Parallelization |
|----------------|---------------------|--------------|-----------------|
| **Python/NumPy** | 1x (baseline) | High | Limited (vectorized ops) |
| **C CPU** | ~10-50x faster | Moderate | OpenMP threads |
| **CUDA GPU** | ~100-500x faster | Optimized | Thousands of threads |

### Key Performance Insights

**Python Implementation:**
- Easy to prototype and debug
- Limited by Python's GIL and interpreted nature
- Suitable for algorithm validation and small grids

**C CPU Implementation:**
- Significant performance improvement over Python
- Efficient memory management
- Multi-threading capabilities with OpenMP

**CUDA GPU Implementation:**
- Massive parallel acceleration
- Optimized memory access patterns using shared memory
- Optimal for production simulations on large grids

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
- **Comprehensive Benchmarking**: Performance comparison across Python, C++, and CUDA
- **Optimized Memory Access**: Shared memory utilization for spatial derivatives
- **Configurable Parameters**: Adjustable grid sizes, diffusion coefficients, and time stepping
- **Scalable Design**: Efficient performance from small test cases to large production runs

## Benchmarking Environment

All performance benchmarks were conducted on:
- **Platform**: Kaggle Notebooks
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **CPU**: Intel Xeon (4 cores)
- **Memory**: 13GB RAM

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
- Computational Fluid Dynamics: Principles and Applications

---

*Developed as part of advanced computational physics coursework focusing on GPU-accelerated numerical simulations.*