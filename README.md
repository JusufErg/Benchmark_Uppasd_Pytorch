# Benchmarking Pytorch Optimizer vs UppASD on the Atomistic Spin Hamiltonian. 

## Introduction

This repository provides benchmarking framework for testing various AI-based optimizers in the context of atomistic spin dynamics simulations, with a focus on skyrmionic spin textures in 2D systems. The project integrates results from the UppASD simulation suite and compares them to optimized spin configurations generated using PyTorch-based implementations in of the atomistic spin Hamiltonian.  

## Theory

### Atomistic Spin Hamiltonian

The physical system modeled in this project is described by the **atomistic spin Hamiltonian**, capturing the interactions between localized magnetic moments on a lattice. The total energy is given by:


![Hamiltonian](https://latex.codecogs.com/png.image?\dpi{150}&space;\mathcal{H}=-\sum_{i\ne&space;j}J_{ij}\,\mathbf{S}_i\cdot\mathbf{S}_j-\sum_{i\ne&space;j}\mathbf{D}_{ij}\cdot(\mathbf{S}_i\times\mathbf{S}_j)-\sum_i&space;K_i(\mathbf{S}_i\cdot\hat{n}_i)^2-\sum_i\mathbf{B}_{\text{ext}}\cdot\mathbf{S}_i)

Where:

- $\mathbf{S}_i$ is the spin (unit vector) at site $i$
- $J_{ij}$ is the **Heisenberg exchange** interaction between spins $i$ and $j$
- $\mathbf{D}_{ij}$ is the **Dzyaloshinskii–Moriya interaction (DMI)** vector
- $K_i$ is the **uniaxial anisotropy** constant and $\hat{n}_i$ is the easy-axis direction
- $\mathbf{B}_{\text{ext}}$ is the **external magnetic field**

Each term corresponds to a distinct physical mechanism:

- **Heisenberg Exchange**: Promotes parallel (ferromagnetic) or antiparallel (antiferromagnetic) spin alignment.
- **DMI**: Introduces chiral interactions that stabilize topological structures like skyrmions.
- **Anisotropy**: Encodes material-specific energy penalties for spin directions.
- **External Field**: Aligns spins along the field direction and can drive transitions between spin textures.

This Hamiltonian governs a complex energy landscape. Optimizing it requires numerical methods capable of navigating such spaces efficiently — motivating the use and benchmarking of modern AI-based optimizers. 

For more information see UppASD webpage/GitHub repository at https://uppasd.github.io/UppASD-tutorial/ 

### Optimizers

In this project, we benchmark a suite of first- and second-order optimization algorithms commonly used in machine learning for minimizing the atomistic spin Hamiltonian. Each optimizer presents different trade-offs in convergence speed, stability, and compute resource usage. These differences are particularly important when dealing with high-dimensional, nonlinear energy landscapes such as those associated with skyrmion lattices.

#### Why Compare Optimizers?

- **Physical Relevance**: Skyrmionic spin textures emerge as local or global minima of the Hamiltonian. Optimizers may differ in their ability to reach physically meaningful configurations.
- **Benchmarking AI4Science Tools**: This work evaluates optimizers in a real scientific context—important for validating their performance beyond standard ML tasks.
- **Hardware-Aware Analysis**: Different optimizers utilize CPU and GPU resources differently. Understanding these trade-offs is essential for scalable simulation.

#### Optimizers Tested

| Optimizer  | Type                      | Highlights                                                                 | Hardware Preference |
|------------|---------------------------|---------------------------------------------------------------------------|---------------------|
| **SGD**     | First-order               | Lightweight and simple; sensitive to learning rate; slower convergence     | CPU or small GPU    |
| **Adam**    | First-order, adaptive     | Fast and robust; combines momentum and adaptive step sizing               | GPU-accelerated     |
| **AdamW**   | First-order, adaptive     | Variant of Adam with better weight decay behavior                         | GPU-accelerated     |
| **RMSprop** | First-order, adaptive     | Stable on noisy gradients; suited for non-stationary problems             | GPU-accelerated     |
| **Adagrad** | First-order, adaptive     | Effective for sparse gradients; decaying learning rate can slow progress  | CPU-compatible      |
| **L-BFGS**  | Quasi-Newton (2nd-order)  | Uses curvature info; fast convergence on smooth loss landscapes; memory-heavy | CPU-preferred   |

#### CPU vs. GPU: Why It Matters

- **First-order optimizers** (like Adam, RMSprop) perform best on **GPUs** due to highly parallelizable operations.
- **Second-order optimizers** (like L-BFGS) are often more **CPU-efficient**, especially for smaller systems or where GPU memory is limited.
- This benchmark includes **runtime comparisons** to evaluate **both accuracy and computational performance** across hardware types.

Understanding these relationships is critical for scaling up simulations and choosing the right optimization strategy in physics-informed machine learning workflows.


## File description

This project is organized into modular Python files that handle parsing, modeling, optimization, evaluation, and benchmarking of spin configurations. Below is a summary of each file, along with its key inputs and outputs.

| File                | Inputs                                                             | Outputs                                                        |
|---------------------|---------------------------------------------------------------------|----------------------------------------------------------------|
| `parser.py`         | `jij`, `dmdata`, `restart.SCsuf_T.out`                              | `data/parsed_restart.csv`                                      |
| `hamiltonian.py`    | Spin configuration tensor, exchange & DMI data, anisotropy, field  | Energy (scalar/tensor), Gradient                               |
| `optimizer.py`      | Initial spin config, Hamiltonian function, optimizer name          | Optimized config, loss trajectory, logs                        |
| `compare_spins.py`  | `optimized_spins.csv`, `parsed_restart.csv`                        | Cosine similarity, angle deviation values/plots                |
| `benchmark_runner.py`| Optimizer name, random seed                                       | Log JSON, trajectory CSV, result dict                          |
| `batch_benchmark.py`| List of optimizers, seed range                                     | Summary plots, log files, aggregate metrics                    |

> Output files are stored in `/data` by default. Parsed spin states from UppASD are saved as `parsed_restart.csv` and serve as the benchmark reference.

### Generated Plots

The benchmarking process produces several visualizations that compare the performance of different optimizers on the spin Hamiltonian. These plots are stored in `/data` and include:

- **`SkyrmionTest_mean_site_diff_boxplot.png`**  
  Displays the distribution of the average per-site angular deviation (in radians or degrees) between each optimizer's result and the UppASD reference configuration. Lower values indicate closer agreement with the true spin texture.

- **`SkyrmionTest_max_site_diff_boxplot.png`**  
  Shows the maximum deviation for any single spin site across trials. Useful for identifying worst-case performance and stability outliers for each optimizer.

- **`SkyrmionTest_final_energy_boxplot.png`**  
  Compares the final minimized energy achieved by each optimizer. This assesses how effectively each optimizer can reach a low-energy configuration, ideally matching the true physical minimum.

- **`SkyrmionTest_timing_plot.png`**  
  Shows the total wall-clock time per optimizer across multiple seeds. This plot helps compare the computational efficiency of each method and their suitability for GPU vs. CPU execution.


## Use

### 1. Prerequisites

- Python ≥ 3.8
- PyTorch
- NumPy, Pandas, Matplotlib

You also need a working installation of **UppASD** to generate simulation data.

> **Note:** You must copy the `SkyrmionLattice` folder from the UppASD distribution into your local directory.

To do this from the command line:

```bash
cp -r /path/to/UppASD/examples/SkyrmionLattice ./SkyrmionLattice
```

For more information on how the UppASD code works, see the link to its webpage above. 

### Directory structure

benchmark_project/
├── SkyrmionLattice/           # Must be copied from UppASD
├── batch_benchmark.py
├── benchmark_runner.py
├── compare_spins.py
├── data/                      # Stores parsed data and results
├── hamiltonian.py
├── optimizer.py
├── parser.py

### Parsing

First you need to run your UppASD simulation, afterwhich you parse the files with the command line

```bash
python parser.py
```

Make sure the parsed_restar.csv file has been generated and is under /data files. 

### Run the benchmarking code

To run for a single or few optimizer simply run the command line

```bash
python batch_benchmark.py --adam --adamw etc.
```

To run all the optimizer simply run 

```bash
python batch_benchmark.py
```


## Results



## Acknowledgements

I would like to express my sincere gratitude to **Digital Futures** and **KTH** for hiring me and providing this valuable internship opportunity. In particular, I would like to thank **Anna Delin**, my main supervisor, for taking me under her mentorship and guiding me through my first steps into real academia. I am also grateful to **Qichen Xu**, my assistant supervisor, for his support and help with technical questions throughout the project.

### Use of AI

I would also like to acknowledge the use of **AI tools (specifically ChatGPT)** during the course of this project. These tools were utilized for various purposes, including debugging, outlining code, refining language, and assisting in the discussion of results.

In today’s technological landscape, claiming no AI involvement in academic work may seem, at best, questionable and, at worst, disingenuous. I believe in embracing this new era with transparency and integrity. By clearly stating how AI contributed to this work, I hope to help set a precedent for openness and help foster a culture of honesty in academia. With that being said, trust goes two ways. 

## Licence 

