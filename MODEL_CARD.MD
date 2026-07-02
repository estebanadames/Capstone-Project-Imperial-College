# Model Card for the Black-Box Optimisation (BBO) Surrogate Models

## Model Details

### Model Name

Bayesian Optimization with Gaussian Process Regression (GPR)

### Project

Imperial College Business School Executive Education

Black-Box Optimisation (BBO) Capstone Project

### Author

Esteban Adames

### Version

Final Capstone Submission (13 Optimisation Rounds)

---

# Model Overview

This project implements a Bayesian Optimization framework designed to optimize eight unknown black-box objective functions under a limited evaluation budget.

Because the mathematical form of each function is unknown, Gaussian Process Regression (GPR) models are used as surrogate models to estimate objective function values and quantify predictive uncertainty. These predictions are combined with an Expected Improvement (EI) acquisition function to recommend promising candidate points for future evaluations.

Independent surrogate models are trained for each of the eight functions and updated after every optimisation round using all historical observations collected up to that point.

The complete optimisation workflow spans thirteen sequential rounds, progressively refining the search toward high-performing regions of each search space.

---

# Intended Use

This optimisation framework is intended for:

- Black-box optimisation problems.
- Expensive objective function optimisation.
- Bayesian Optimization research.
- Sequential experimental design.
- Hyperparameter optimisation.
- Machine Learning education.
- Engineering optimisation problems.
- Demonstrating surrogate modelling techniques.

The project is primarily educational and demonstrates how Bayesian Optimization can efficiently solve optimisation problems when objective function evaluations are expensive or unavailable analytically.

---

# Users

This repository is intended for:

- Machine Learning students.
- Data Scientists.
- Optimization researchers.
- Engineers working on expensive optimisation problems.
- Academic instructors.
- Employers evaluating machine learning portfolios.

---

# Training Data

The surrogate models are trained using the complete optimisation history collected during the capstone project.

The dataset includes:

- Historical query coordinates.
- Corresponding objective function evaluations.
- Sequential observations collected across thirteen optimisation rounds.

Each optimisation round incorporates all previous observations before retraining the surrogate model.

Function dimensionalities:

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

The dataset contains no personal or sensitive information.

---

# Model Architecture

The optimisation framework consists of:

1. Historical query dataset.
2. Gaussian Process Regression surrogate model.
3. Matern covariance kernel.
4. White noise kernel.
5. Hyperparameter optimisation through maximum likelihood estimation.
6. Expected Improvement acquisition function.
7. Sequential query generation.

Each objective function is modelled independently.

---

# Training Procedure

Each optimisation round followed the same workflow:

1. Load all historical observations.
2. Train a Gaussian Process Regression model.
3. Optimize kernel hyperparameters.
4. Predict objective values across candidate points.
5. Evaluate Expected Improvement.
6. Select the highest-scoring candidate.
7. Submit the candidate to the optimisation platform.
8. Append the returned evaluation to the dataset.

This process was repeated throughout thirteen optimisation rounds.

---

# Performance

The optimisation process demonstrated steady improvement across several objective functions as additional observations became available.

Final best observed objective values:

| Function | Best Output |
|----------|------------:|
| Function 1 | 0.000000 |
| Function 2 | 0.714674 |
| Function 3 | -0.012631 |
| Function 4 | 0.675598 |
| Function 5 | 5920.981579 |
| Function 6 | -0.134276 |
| Function 7 | 1.990473 |
| Function 8 | 9.990223 |

Functions 5, 7 and 8 exhibited strong convergence toward high-performing regions.

Functions 3 and 6 remained more challenging due to their more complex optimisation landscapes.

---

# Evaluation Metrics

Because the true objective functions remain hidden, conventional supervised learning metrics (such as accuracy or RMSE) cannot be computed.

Model performance was instead evaluated using:

- Improvement in objective function value.
- Sequential optimisation performance.
- Convergence toward promising regions.
- Sample efficiency.
- Stability across optimisation rounds.

---

# Strengths

The optimisation framework offers several advantages:

- Sample-efficient optimisation.
- Explicit modelling of predictive uncertainty.
- Effective balance between exploration and exploitation.
- Robust performance with relatively few observations.
- Fully reproducible optimisation workflow.

The iterative learning process allowed increasingly informed decisions as more observations became available.

---

# Limitations

Several limitations should be acknowledged:

- The Gaussian Process assumes relatively smooth objective functions.
- Performance depends heavily on the quality and diversity of sampled observations.
- Sequential sampling introduces adaptive sampling bias.
- Some regions of the search space remain unexplored due to the limited evaluation budget.
- Bayesian Optimization cannot guarantee discovery of the true global optimum within a finite number of evaluations.
- High-dimensional optimisation becomes increasingly difficult as dimensionality grows.

These limitations are common in expensive black-box optimisation problems.

---

# Ethical Considerations

This project operates exclusively on synthetic optimisation data and therefore does not directly affect individuals or sensitive populations.

However, Bayesian Optimization techniques are increasingly applied in domains such as healthcare, finance, manufacturing and scientific experimentation. In those contexts, optimisation results should always be validated using domain expertise, fairness assessments and rigorous testing before deployment.

This repository is intended for educational purposes and demonstrates transparent, reproducible machine learning practices.

---

# Reproducibility

The project was developed to maximize reproducibility.

The repository contains:

- Weekly optimisation scripts.
- Historical datasets.
- Query histories.
- Documentation.
- README.
- Datasheet.
- Model Card.

All optimisation rounds are version controlled using Git and GitHub, allowing the complete evolution of the optimisation strategy to be reproduced and audited.

---

# Future Improvements

Potential future enhancements include:

- Upper Confidence Bound (UCB) acquisition functions.
- Probability of Improvement (PI).
- Ensemble surrogate models.
- Multi-start acquisition optimisation.
- Adaptive kernel selection.
- Deep Gaussian Processes.
- Reinforcement Learning–guided exploration.
- Batch Bayesian Optimization for parallel evaluations.

These extensions could improve convergence speed and robustness for more complex optimisation problems.

---

# References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.*
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). *Efficient Global Optimization of Expensive Black-Box Functions.*
- Imperial College Business School Executive Education – Black-Box Optimisation Capstone Project.