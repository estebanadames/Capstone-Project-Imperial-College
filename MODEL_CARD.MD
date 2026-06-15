# Model Card: BBO Gaussian Process Optimisation Strategy

## Overview

### Model Name

BBO Gaussian Process Optimisation Framework

### Version

v1.0 (Rounds 1–10)

### Model Type

Sequential surrogate-based optimisation using:

* Gaussian Process Regression (GPR)
* Matern Kernel
* White Noise Kernel
* Expected Improvement acquisition function

Implemented using Python and scikit-learn.

---

## Intended Use

### Suitable Uses

* Black-box optimisation
* Expensive function evaluation problems
* Low-data optimisation environments
* Sequential experiment design

### Unsuitable Uses

* Large-scale deep learning optimisation
* Real-time control systems
* High-dimensional problems requiring thousands of evaluations

---

## Details

The optimisation process evolved over ten rounds.

### Early Rounds

* Broad exploration
* Randomised query diversity
* Initial surrogate construction

### Middle Rounds

* Increased reliance on Gaussian Process predictions
* Expected Improvement used to balance exploration and exploitation
* Identification of promising regions

### Later Rounds

* Focused refinement around high-performing areas
* Reduced exploratory behaviour
* Greater use of accumulated historical data

The approach continuously retrained the Gaussian Process model after each new observation.

---

## Performance

Performance was evaluated using:

* Best observed objective value
* Improvement over previous rounds
* Convergence behaviour

Notable results:

| Function   | Best Observed Output |
| ---------- | -------------------- |
| Function 1 | ~0                   |
| Function 2 | 0.611                |
| Function 3 | -0.013               |
| Function 4 | 0.629                |
| Function 5 | 5638.809             |
| Function 6 | -0.183               |
| Function 7 | 1.886                |
| Function 8 | 9.986                |

Function 5, Function 7 and Function 8 demonstrated substantial improvement over successive rounds.

---

## Assumptions and Limitations

### Assumptions

* Objective functions are smooth enough to be modelled by a Gaussian Process.
* Nearby inputs have correlated outputs.
* Historical observations remain representative of future behaviour.

### Limitations

* Small sample size.
* Potential local optima.
* Reduced reliability in higher-dimensional spaces.
* Sensitivity to kernel hyperparameters.

Performance may degrade if the underlying function violates Gaussian Process assumptions.

---

## Ethical Considerations

Transparency improves reproducibility and accountability.

All optimisation decisions are documented through:

* Historical queries
* Recorded outputs
* Source code
* Repository documentation

The project does not use personal data, demographic data, or sensitive information.

The primary ethical objective is methodological transparency, allowing other researchers to reproduce and evaluate the optimisation process.

---

## Future Improvements

Potential future enhancements include:

* Ensemble surrogate models
* Neural-network-based surrogates
* Bayesian neural networks
* Multi-objective optimisation
* Adaptive kernel selection
* Hyperparameter optimisation frameworks

