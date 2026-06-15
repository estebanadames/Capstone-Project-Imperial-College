# Datasheet for the BBO Capstone Dataset

## Motivation

This dataset was created as part of the Imperial College Business School Black-Box Optimisation (BBO) Capstone Project.

The objective is to optimise eight unknown functions using a limited number of evaluations. The dataset supports experimentation with optimisation algorithms, surrogate modelling, Bayesian optimisation, and sequential decision-making under uncertainty.

The dataset was created to track query histories, function evaluations, and optimisation performance across multiple rounds of experimentation.

---

## Composition

The dataset contains:

* Query inputs submitted to eight black-box functions
* Corresponding function outputs
* Historical query records across ten optimisation rounds

Each function has a fixed dimensionality:

| Function   | Dimensions |
| ---------- | ---------- |
| Function 1 | 2          |
| Function 2 | 2          |
| Function 3 | 3          |
| Function 4 | 4          |
| Function 5 | 4          |
| Function 6 | 5          |
| Function 7 | 6          |
| Function 8 | 8          |

By Round 10, the dataset contained approximately 80 query-output observations.

Data is stored in NumPy array format and Python scripts.

No personal, demographic, or sensitive information is included.

---

## Collection Process

The data was collected through iterative interaction with the BBO challenge platform.

Each week:

1. Previous inputs and outputs were analysed.
2. A Gaussian Process surrogate model was fitted.
3. Expected Improvement was used as the acquisition strategy.
4. A new candidate query was generated.
5. The query was submitted to the challenge platform.
6. The returned function evaluation was added to the dataset.

Data collection occurred over ten optimisation rounds.

---

## Preprocessing, Cleaning and Labelling

Minimal preprocessing was required.

Steps included:

* Converting query values into NumPy arrays
* Standardising numerical precision
* Organising results by optimisation round
* Storing inputs and outputs in structured Python files

No labels were manually assigned.

No missing values were encountered.

---

## Uses

### Intended Uses

* Bayesian optimisation research
* Surrogate modelling experiments
* Sequential optimisation studies
* Machine learning education
* Black-box optimisation benchmarking

### Inappropriate Uses

* Real-world decision-making without validation
* Performance comparisons against domains outside black-box optimisation
* Applications involving human subjects or sensitive data

---

## Distribution

The dataset is available through the project's public GitHub repository.

Distribution is limited to educational and research purposes associated with the capstone project.

No licensing restrictions beyond the repository license apply.

---

## Maintenance

The dataset is maintained by the project author.

Updates occur whenever new optimisation rounds are completed.

Version control is managed through GitHub to ensure transparency and reproducibility.
