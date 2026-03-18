# Fuzzy-SPIKAN

**Official implementation of "Fuzzy-SPIKANs: Enhancing Separable Physics-Informed Kolmogorov-Arnold Networks via Continuous Fuzzy Aggregation"**

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![JAX](https://img.shields.io/badge/JAX-0.6.2-blue.svg)](https://github.com/google/jax)

## Overview

Separable Physics-Informed Kolmogorov-Arnold Networks (SPIKANs) efficiently mitigate the curse of dimensionality in multidimensional PDE solvers. However, their reliance on linear summation for node aggregation creates an **"additive bottleneck,"** diluting high-frequency features and mutually exclusive states in non-linear physical systems.

**Fuzzy-SPIKAN** overcomes this limitation by integrating **generalized continuous fuzzy logic operators (Fuzzy OR and Generalized XOR)** into the separable architecture. To prevent gradient explosion in deep spatiotemporal domains, we introduce **logic-preserving scaled squashing functions** (scaled tanh/sigmoid) that strictly maintain exact fuzzy boundaries.

Furthermore, this continuous fuzzy aggregation serves as a **universal, plug-and-play module** that can be seamlessly integrated into any general KAN or PINN architecture.

### Key Highlights
- **Mathematical Rigor:** Establishes the Universal Approximation Theorem (UAT) for continuously squashed fuzzy aggregations.
- **Robustness:** Prevents trivial solution collapse in highly stiff dynamics (e.g., Allen-Cahn equation).
- **High Accuracy:** Achieves up to a 24-fold error reduction in complex spatiotemporal wave propagation (e.g., Klein-Gordon equation).

---

## Installation

The code is written in Python and relies on JAX for high-performance automatic differentiation. 

1. Clone this repository:

git clone [https://github.com/ejhwang312/Fuzzy_SPIKAN.git](https://github.com/ejhwang312/Fuzzy_SPIKAN.git)
cd Fuzzy_SPIKAN

2. Install the required dependencies:

pip install -r requirements.txt

Note: For GPU acceleration, please ensure you install the correct version of jaxlib that matches your CUDA version. See the official JAX installation guide for more details.


* Acknowledgements

This repository is built upon the official implementation of SPIKAN (Licensed under BSD 2-Clause, Copyright 2025 Battelle Memorial Institute). We express our sincere gratitude to the original authors for making their code publicly available.

Our main modifications to accommodate the Fuzzy-SPIKAN framework include:

Replacing the linear node aggregation with continuous Fuzzy OR/XOR operators.

Introducing logic-preserving scaled squashing functions to prevent gradient explosion.

This work was supported by the Global - Learning & Academic research institution for Master's/PhD students, and Postdocs (G-LAMP) Program of the National Research Foundation of Korea (NRF) grant funded by the Ministry of Education (No. RS-2025-25442252).


