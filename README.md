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
| `compare_spins.py`  | `optimized_spins.csv`, `parsed_restart.csv`                        | Cosine similarity, angle deviation values/plots, in degrees                |
| `benchmark_runner.py`| Optimizer name, random seed                                       | Log JSON, trajectory CSV, result dict                          |
| `batch_benchmark.py`| List of optimizers, seed range                                     | Summary plots, log files, aggregate metrics                    |

> Output files are stored in `/data` by default. Parsed spin states from UppASD are saved as `parsed_restart.csv` and serve as the benchmark reference.
> 
The SkyrmionLattice files are the specific input files used for the UppASD simulation, included here to allow for reproducibility of the results.

### Generated Plots

The benchmarking process produces several visualizations that compare the performance of different optimizers on the spin Hamiltonian. These plots are stored in `/data` and include:

- **`SkyrmionTest_Mean per-site angular deviation (°)_boxplot.png`**  
  Displays the distribution of the average per-site angular deviation (in degrees) between each optimizer's result and the UppASD reference configuration. Lower values indicate closer agreement with the true spin texture.

- **`SkyrmionTest_Max per-site angular deviation (°)_boxplot.png`**  
  Shows the maximum deviation for any single spin site across trials. Useful for identifying worst-case performance and stability outliers for each optimizer.

- **`SkyrmionTest_final_energy_boxplot.png`**  
  Compares the final minimized energy achieved by each optimizer. This assesses how effectively each optimizer can reach a low-energy configuration, ideally matching the true physical minimum.

- **`SkyrmionTest_timing_plot.png`**  
  Shows the total wall-clock time per optimizer across multiple seeds. This plot helps compare the computational efficiency of each method and their suitability for GPU vs. CPU execution.

### Output Files and Their Descriptions

After running the benchmark pipeline, the following files will appear in the `data/` directory:

#### Spin Configurations
- `parsed_restart.csv`  
  Parsed from `restart.<simid>.out` (UppASD output). Contains the initial spin configuration.  
  **Columns:** `site, atom, mx, my, mz`

- `optimized_spins_<simid>_<optimizer>.csv`  
  Final spin configuration after optimization with the specified optimizer.  
  One file per optimizer and run.  
  **Columns:** `mx, my, mz, atom` (indexed by site)

---

#### Energy Logs
- `energy_log_<simid>_<optimizer>.csv`  
  Records energy values at each optimization step.  
  **Columns:** `step, total, heisenberg, dmi, anisotropy`

---

#### Benchmark Summaries
- `benchmark_summary_<simid>_runN.csv`  
  Summary for each run, including final energy and angular deviations.

- `batch_summary_<simid>.csv`  
  Combined summary across all runs and optimizers.  
  **Columns:** `optimizer, final_energy, mean_angle, max_angle, run`

---

#### Timing
- `timing_log.csv`  
  Records wall-clock time for each optimizer per run.  
  **Columns:** `optimizer, run, time (seconds)`


These plots summarize the performance and consistency of each optimizer across runs.

### The Results Files

The final two files in the repository are zipped archives containing all the benchmarking results. These were compressed because their expanded size exceeded GitHub's upload limit. When expanded you will see further files named `results_<a number>_<point either 1 or 01>`. 

The `results_*_*` files store benchmark data from runs with different external magnetic fields (`hfield`) and learning rates.

The naming convention is:

- `results_<z>_<lr>`  
  where:
  
  - `<z>` is the negative z-component of the applied external magnetic field (`hfield = [0, 0, -z]`).  
    - For example, `results_75_*` corresponds to `hfield = [0, 0, -75]`, and `results_1_*` to `hfield = [0, 0, -1]`.
    - The values tested were -5, -1, -0.1, and 0. 
  
  - `<lr>` encodes the learning rate:
    - `point1` refers to `0.1`
    - `point01` refers to `0.01`
  
Thus:
- `results_75_point1` → `hfield = [0, 0, -75]`, `lr = 0.1`
- `results_1_point01` → `hfield = [0, 0, -1]`, `lr = 0.01`

These results were generated on the **Dardel supercomputer** at KTH, Sweden, using the following parameters:

```python
LR = 0.1 or 0.01
STEPS = 1000
SIMID = "SkyrmionTest"
RESTART_CSV = "data/parsed_restart.csv"
RUNS = 10
```

You can find detailed explanations and discussion of these results under the **Results** section below.

## Project Particulars

Certain choices in this project were made based on both practical limitations and physical reasoning:

- **Anisotropy Term Omitted**  
  The SkyrmionLattice example used from UppASD does not specify the **uniaxial anisotropy term** ($K_i (\mathbf{S}_i \cdot \hat{n}_i)^2$) in the Hamiltonian. As a result, this term is omitted during optimization to remain consistent with the input configuration. This simplification avoids introducing mismatches between the reference and modelled systems.

- **External Field Modified**  
  The external magnetic field term ($\mathbf{B}_{\text{ext}} \cdot \mathbf{S}_i$) was modified in our experiments. While the default in UppASD is often set to zero, we explored different non-zero values of `hfield` to test how well each optimizer handles larger energy scales. The aim was to examine whether stronger external fields would degrade optimizer performance or overpower the effects of DMI and exchange.

Further discussion and analysis of these experiments are provided in the [Results](#results) section below.

> **Note:** The specific UppASD input files used in this project are included in the `SkyrmionLattice/` folder (ie. momfile, qfile, inpsd.dat, dmidata, posfile, jij). 

## Use

### 1. Prerequisites

- Python ≥ 3.8
- PyTorch
- NumPy, Pandas, Matplotlib

> **Note** specifically, when the code was written these were the versions used: Python 3.9.23, [Clang 14.0.6 ], Torch 2.2.2, Pandas 2.3.0+4.g1dfc98e16a, NumPy 1.26.4

You also need a working installation of **UppASD** to generate simulation data.

> **Note:** You must copy the `SkyrmionLattice` folder from the UppASD distribution into your local directory.

To do this from the command line:

```bash
cp -r /path/to/UppASD/examples/SkyrmionLattice ./SkyrmionLattice
```

For more information on how the UppASD code works, see the link to its webpage above. 

### Directory structure

<pre>
benchmark_project/
├── SkyrmionLattice/           # Must be copied from UppASD
├── batch_benchmark.py
├── benchmark_runner.py
├── compare_spins.py
├── data/                      # Stores parsed data and results
├── hamiltonian.py
├── optimizer.py
├── parser.py
</pre>

### Parsing UppASD Output
First run your UppASD simulation. Afterward, parse the output with:

```bash
python parser.py
```

This reads files like restart.SCsuf_T.out from SkyrmionLattice/ and saves the result to:

```
data/parsed_restart.csv
```

Make sure this file exists before proceeding to benchmarking.

### Running the Benchmarking Code
To run benchmarking for specific optimizers, use:

```bash
python batch_benchmark.py --adam --adamw --sgd
```

To run all optimizers with default settings:

```bash
python batch_benchmark.py
```

Results (logs, optimized spins, plots) will be saved in the /data directory.

You can change how many runs you want the code to do in batch_benchmark.py by changing the line 

```
RUNS = ...
```


### Comparing Against UppASD Result
To compare an optimized spin configuration to the UppASD reference (parsed_restart.csv), use:

```bash
python compare_spins.py --file path/to/your/spins.csv
```

This outputs:

Mean angular deviation (how far each spin tilts, on average)
Maximum per-site angular deviation (worst-case spin error)


## Results

### output files

## Acknowledgements

I would like to express my sincere gratitude to **Digital Futures** and **KTH** for hiring me and providing this valuable internship opportunity. In particular, I would like to thank **Anna Delin**, my main supervisor, for taking me under her mentorship and guiding me through my first steps into real academia. I am also grateful to **Qichen Xu**, my assistant supervisor, for his support and help with technical questions throughout the project.

### Use of AI

I would also like to acknowledge the use of **AI tools (specifically ChatGPT)** during the course of this project. These tools were utilized for various purposes, including debugging, outlining code, refining language, and assisting in the discussion of results.

In today’s technological landscape, claiming no AI involvement in academic work may seem, at best, questionable and, at worst, disingenuous. I believe in embracing this new era with transparency and integrity. By clearly stating how AI contributed to this work, I hope to help set a precedent for openness and help foster a culture of honesty in academia. With that being said, trust goes two ways. 

## References



## License

This project is released under the [MIT License](LICENSE).

Note: This repository depends on the UppASD simulation suite, which is not included and must be obtained separately from UppASD's official website. Please respect their licensing terms when using or redistributing simulation data.

