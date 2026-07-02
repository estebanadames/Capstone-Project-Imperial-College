# Datasheet for the Black-Box Optimisation (BBO) Capstone Dataset

## Motivation

This datasheet documents the dataset developed throughout the **Imperial College Business School Executive Education – Black-Box Optimisation (BBO) Capstone Project**.

The objective of the project was to optimize eight unknown black-box objective functions under a strict evaluation budget. Since the analytical form of each function was unavailable, the optimization process relied on sequential observations collected over multiple rounds to train surrogate models and guide future query selections.

This dataset captures the complete optimization history across thirteen iterative rounds and supports experimentation with Bayesian Optimization, Gaussian Process Regression, surrogate modelling and sequential decision-making under uncertainty.

The primary purpose of documenting this dataset is to promote transparency, reproducibility and responsible machine learning practices.

---

# Dataset Composition

The dataset contains:

- Historical query vectors submitted to eight black-box objective functions.
- Corresponding objective function evaluations returned by the optimization platform.
- Sequential observations collected across thirteen optimisation rounds.
- Weekly optimisation histories used to retrain surrogate models.

Each objective function has a fixed dimensionality:

| Function | Dimensions |
|----------|-----------:|
| Function 1 | 2 |
| Function 2 | 2 |
| Function 3 | 3 |
| Function 4 | 4 |
| Function 5 | 4 |
| Function 6 | 5 |
| Function 7 | 6 |
| Function 8 | 8 |

Across thirteen rounds, the dataset contains more than one hundred query–output observations distributed across the eight optimisation problems.

Data is stored as Python scripts and NumPy arrays.

The dataset contains **no personal, demographic, financial or sensitive information**.

---

# Data Collection Process

The dataset was generated through iterative interaction with the Imperial College BBO challenge platform.

Each optimisation round followed the same workflow:

1. Historical observations from previous rounds were collected.
2. A Gaussian Process Regression surrogate model was trained independently for each function.
3. An Expected Improvement acquisition function identified promising candidate locations.
4. New query points were generated.
5. Queries were submitted to the challenge platform.
6. Objective function evaluations were returned.
7. The new observations were incorporated into the historical dataset for the following optimisation round.

Because the dataset was collected sequentially, every new observation depended on information obtained during previous rounds.

---

# Preprocessing and Data Preparation

Only minimal preprocessing was required because all variables were continuous numerical values.

Preparation steps included:

- Converting query coordinates into NumPy arrays.
- Maintaining six-decimal numerical precision.
- Organising observations by optimisation round.
- Separating inputs and outputs for each function.
- Validating dimensional consistency before model training.

No observations required manual correction.

No missing values were encountered during the optimisation process.

No manual labels were assigned.

---

# Intended Uses

This dataset is intended for:

- Bayesian Optimization research.
- Gaussian Process Regression experiments.
- Surrogate modelling studies.
- Sequential optimisation research.
- Educational demonstrations of expensive black-box optimisation.
- Machine Learning coursework and reproducible research.

The dataset is particularly appropriate for illustrating optimisation under limited evaluation budgets.

---

# Uses That Should Be Avoided

This dataset should **not** be used for:

- Benchmarking unrelated optimisation algorithms without considering the limited evaluation budget.
- Drawing conclusions about real-world engineering systems.
- Applications involving healthcare, finance or other high-stakes domains without additional validation.
- Statistical inference beyond the scope of the optimisation challenge.

Because the underlying objective functions remain unknown, conclusions should be interpreted within the context of this educational optimisation exercise.

---

# Distribution

The dataset is distributed through this public GitHub repository together with:

- Source code
- Weekly optimisation scripts
- Documentation
- Datasheet
- Model Card
- Project README

The repository is intended for educational, research and portfolio purposes.

---

# Maintenance

The dataset was maintained throughout the capstone project by the project author.

Version control was managed using Git and GitHub, allowing every optimisation round to be tracked transparently.

No further observations are expected following completion of the final optimisation round; however, the repository will remain available as a reproducible record of the project methodology.

---

# Limitations

Several limitations should be considered when using this dataset:

- The dataset is relatively small because each optimisation round permitted only one query per function.
- Sampling is adaptive rather than random, introducing bias toward promising regions of the search space.
- The true analytical form of the objective functions is unknown, preventing direct validation of surrogate accuracy.
- Some regions of the search space remain unexplored due to the limited evaluation budget.
- Gaussian Process surrogate models assume a degree of smoothness that may not perfectly represent every objective function.

These limitations are characteristic of many real-world black-box optimisation problems where evaluations are expensive.

---

# Ethical Considerations

The dataset contains only synthetic optimisation observations and does not include information about individuals.

Although Bayesian Optimization techniques are frequently applied in real-world domains such as healthcare, manufacturing and finance, this project is intended solely as an educational demonstration of sequential optimisation methods.

The dataset therefore presents minimal ethical or privacy risks while promoting transparent, reproducible and responsible machine learning practices.