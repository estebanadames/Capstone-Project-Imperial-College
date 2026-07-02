# Black-Box Optimisation (BBO) Capstone Project
---

# Overview

This repository contains my complete work for the **Imperial College Business School Executive Education – Black-Box Optimisation (BBO) Capstone Project**.

The objective of the project is to optimise **eight unknown black-box objective functions** using a limited evaluation budget. Since the analytical expressions of the functions are hidden, each new query must be selected intelligently based only on historical observations.

To solve this challenge, I implemented a **Bayesian Optimization** workflow using **Gaussian Process Regression (GPR)** as a surrogate model together with **Expected Improvement (EI)** to balance exploration and exploitation throughout thirteen optimisation rounds.

The project documents not only the code, but also the evolution of the optimisation strategy, weekly reflections, datasets, model documentation and final results.

---

# Non-Technical Summary

Many real-world problems involve systems that are expensive or impossible to understand mathematically. Instead of knowing exactly how a system behaves, we only observe its inputs and outputs. This project demonstrates how machine learning can intelligently search for better solutions under those conditions.

Using Bayesian Optimization, the algorithm gradually learned which regions of the search space produced better results and adapted its decisions accordingly. Rather than testing random points, each new experiment became increasingly informed by previous observations, allowing the optimisation process to become more efficient over time.

---

# Project Objectives

The primary goals of this project were to:

- Optimize eight unknown objective functions.
- Build surrogate models that learn from limited observations.
- Apply Bayesian Optimization under strict query constraints.
- Balance exploration and exploitation effectively.
- Track optimisation performance across thirteen sequential rounds.
- Develop a reproducible and transparent machine learning workflow.

---

# Dataset

The dataset consists of:

- Historical query locations submitted each week.
- Corresponding objective function evaluations returned by the BBO platform.
- Weekly optimisation history accumulated across thirteen rounds.

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

Each week's observations were incorporated into the historical dataset before training the next surrogate model.

---

# Methodology

The optimisation workflow followed an iterative Bayesian Optimization pipeline.

## 1. Initial Exploration

During the first iterations, query points were intentionally distributed across different regions of the search space to maximize information gain.

---

## 2. Gaussian Process Surrogate Models

After collecting sufficient observations, Gaussian Process Regression models were trained for each function using:

- Matern kernels
- White noise kernels
- Automatic hyperparameter optimisation
- Maximum likelihood estimation

These surrogate models approximate the unknown objective functions while quantifying predictive uncertainty.

---

## 3. Acquisition Function

Expected Improvement (EI) was used to determine where to sample next.

EI balances:

- **Exploitation:** sampling near promising regions.
- **Exploration:** sampling uncertain regions that could contain better optima.

---

## 4. Weekly Optimisation Process

Each optimisation round followed the same workflow:

1. Update the historical dataset.
2. Retrain Gaussian Process models.
3. Evaluate Expected Improvement.
4. Generate candidate query points.
5. Submit new queries.
6. Record outputs.
7. Repeat.

As more observations became available, the optimisation strategy gradually shifted from exploration toward local refinement around high-performing regions.

---

# Repository Structure

```text
BBO_Capstone/
│
├── README.md
├── DATASHEET.md
├── MODEL_CARD.md
├── LICENSE
├── requirements.txt
│
├── initial_data/
│   ├── bbo_week2/
│   ├── bbo_week3/
│   ├── ...
│   ├── bbo_week13/
│
├── function_1/
├── function_2/
├── function_3/
├── function_4/
├── function_5/
├── function_6/
├── function_7/
├── function_8/
│
├── bbo_week2_gp.py
├── bbo_week3_gp.py
├── ...
├── bbo_week13_gp.py
```

---

# Technologies Used

## Programming Language

- Python 3

## Core Libraries

- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib

## Machine Learning Methods

- Bayesian Optimization
- Gaussian Process Regression
- Surrogate Modelling
- Expected Improvement
- Sequential Decision Making

---

# Results

Across thirteen optimisation rounds, the surrogate models became increasingly accurate as more observations were incorporated.

The largest improvements were observed in:

| Function | Best Score |
|----------|-----------:|
| Function 1 | 0.000000 |
| Function 2 | 0.714674 |
| Function 3 | -0.012631 |
| Function 4 | 0.675598 |
| Function 5 | 5920.981579 |
| Function 6 | -0.134276 |
| Function 7 | 1.990473 |
| Function 8 | 9.990223 |

Functions 5, 7 and 8 demonstrated particularly strong convergence, while Functions 3 and 6 remained more challenging due to their more complex optimization landscapes.

---

# Key Insights

Several important patterns emerged during the project:

- Exploration was most valuable during the early rounds.
- Exploitation became increasingly effective as confidence in the surrogate models improved.
- Different objective functions required different optimisation behaviours.
- Local refinement produced substantial gains once promising regions were identified.
- Small, informed adjustments consistently outperformed random exploration during the later stages.

---

# Documentation

The repository also includes project documentation following current ML transparency practices.

## Datasheet

📄 [DATASHEET.md](DATASHEET.md)

Describes:

- Dataset motivation
- Collection process
- Composition
- Intended uses
- Limitations
- Maintenance

---

## Model Card

📄 [MODEL_CARD.md](MODEL_CARD.md)

Describes:

- Bayesian Optimization methodology
- Model assumptions
- Performance
- Limitations
- Ethical considerations
- Appropriate use cases

---

# Lessons Learned

This capstone provided practical experience in:

- Bayesian Optimization
- Gaussian Process modelling
- Black-box optimisation
- Sequential decision making
- Hyperparameter optimisation
- Exploration versus exploitation
- Reproducible machine learning research

Perhaps the most important lesson was that optimisation is fundamentally an iterative learning process. Rather than searching randomly, every new evaluation should improve the model's understanding of the problem and guide increasingly informed decisions.

---

# Future Improvements

Given additional evaluation budget, future work could include:

- Alternative acquisition functions (Upper Confidence Bound, Probability of Improvement)
- Ensemble surrogate models
- Multi-start acquisition optimisation
- Adaptive exploration schedules
- Reinforcement Learning inspired exploration policies
- Automated kernel selection

---

# Author

**Esteban Adames**

VCA Core Consulting Analyst

Imperial College Business School Executive Education

Black-Box Optimisation (BBO) Capstone Project

2026