# Black-Box Optimisation (BBO) Capstone Project

## Overview

This repository contains my work for the Imperial College Business School Black-Box Optimisation (BBO) Capstone Project.

The objective of the challenge is to optimise eight unknown black-box functions using a limited number of evaluations. Since the analytical form of each function is hidden, the problem requires building surrogate models that learn from historical observations and guide future query selections.

Throughout the project, I developed and refined a Bayesian Optimisation workflow using Gaussian Process Regression and Expected Improvement acquisition functions to balance exploration and exploitation.

---

## Project Objectives

The main goals of this project are:

* Optimise eight unknown objective functions.
* Learn efficient black-box optimisation strategies.
* Evaluate surrogate modelling techniques.
* Balance exploration and exploitation under limited data.
* Track optimisation performance across multiple rounds.

This project simulates real-world optimisation problems where experiments are expensive and objective functions are unknown.

---

## Dataset

The dataset consists of:

* Historical query inputs submitted to each function.
* Corresponding function evaluations returned by the challenge platform.
* Query histories accumulated over ten optimisation rounds.

Function dimensionalities:

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

---

## Methodology

### Initial Exploration

During the first rounds, I explored different regions of the search space to collect diverse observations.

### Surrogate Modelling

After collecting sufficient observations, I trained Gaussian Process Regression (GPR) models using:

* Matern kernels
* White noise kernels
* Hyperparameter optimisation

These surrogate models approximate the unknown objective functions.

### Acquisition Function

Expected Improvement (EI) was used to identify promising candidate points while maintaining exploration of uncertain regions.

### Iterative Optimisation

Each week:

1. Historical observations were collected.
2. Gaussian Process models were retrained.
3. Expected Improvement was evaluated.
4. New candidate queries were generated.
5. Results were incorporated into the dataset.

---

## Repository Structure

```text
BBO_Capstone/
│
├── README.md
├── DATASHEET.md
├── MODEL_CARD.md
│
├── initial_data/
│   ├── bbo week 2/
│   ├── bbo week 3/
│   ├── bbo week 4/
│   ├── bbo week 5/
│   ├── bbo week 6/
│   ├── bbo week 7/
│   ├── bbo week 8/
│   ├── bbo week 9/
│   ├── bbo week 10/
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
├── bbo_week4_gp.py
├── bbo_week5_gp.py
├── bbo_week6_gp.py
├── bbo_week7_gp.py
├── bbo_week8_gp.py
├── bbo_week9_gp.py
├── bbo_week10_gp.py
```

---

## Technologies Used

### Programming Language

* Python 3

### Libraries

* NumPy
* SciPy
* scikit-learn

### Machine Learning Methods

* Gaussian Process Regression (GPR)
* Bayesian Optimisation
* Expected Improvement (EI)
* Surrogate Modelling

---

## Results

By Round 10, the optimisation process had produced substantial improvements in several functions, particularly:

* Function 5
* Function 7
* Function 8

The iterative Bayesian Optimisation approach consistently identified higher-performing regions while reducing unnecessary exploration.

---

## Documentation

### Datasheet

See:

* [Datasheet](DATASHEET.md)

This document describes the dataset composition, collection process, intended uses and maintenance procedures.

### Model Card

See:

* [Model Card](MODEL_CARD.md)

This document explains the optimisation methodology, assumptions, limitations and performance characteristics.

---

## Key Lessons Learned

This project provided practical experience in:

* Bayesian Optimisation
* Surrogate modelling
* Sequential decision making
* Hyperparameter tuning
* Exploration versus exploitation trade-offs
* Reproducible machine learning workflows

It also demonstrated how optimisation problems can be solved efficiently even when the underlying objective functions remain completely unknown.

---

## Author

Esteban Adames

Imperial College Business School Executive Education

Black-Box Optimisation Capstone Project
