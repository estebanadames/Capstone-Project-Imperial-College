import os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

BASE_DIR = "."
FUNCTION_FOLDERS = [f"function_{i}" for i in range(1, 9)]

np.random.seed(42)

new_inputs = [
    np.array([0.500000, 0.500000]),
    np.array([0.786108, 0.766213]),
    np.array([0.509324, 0.672224, 0.305324]),
    np.array([0.250628, 0.593730, 0.622854, 0.050000]),
    np.array([0.242292, 0.890450, 0.872711, 0.880542]),
    np.array([0.683147, 0.544891, 0.407203, 0.817682, 0.076647]),
    np.array([0.241025, 0.683264, 0.609997, 0.833195, 0.173365, 0.391061]),
    np.array([0.213526, 0.434710, 0.283977, 0.510215, 0.869933, 0.671704, 0.264676, 0.265407]),
]

new_outputs = np.array([
    2.6752879910742468e-09,
    0.03838400221215603,
    -0.09560907240309474,
    -11.923018749116093,
    1275.2685405563973,
    -0.7363719843000791,
    0.1660141105990867,
    9.6407935213586
])

def expected_improvement(X, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-12)

    improvement = mu - y_best - xi
    z = improvement / sigma

    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

def format_query(x):
    return "-".join(f"{v:.6f}" for v in x)

def main():
    print("\nWeek 2 GP Query Suggestions")
    print("-" * 50)

    for i, func_name in enumerate(FUNCTION_FOLDERS):
        folder_path = os.path.join(BASE_DIR, func_name)

        X_old = np.load(os.path.join(folder_path, "initial_inputs.npy"))
        y_old = np.load(os.path.join(folder_path, "initial_outputs.npy"))

        X = np.vstack([X_old, new_inputs[i]])
        y = np.append(y_old, new_outputs[i])

        d = X.shape[1]

        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        gp.fit(X, y)

        X_candidates = np.random.uniform(0, 1, size=(5000, d))
        y_best = np.max(y)

        ei = expected_improvement(X_candidates, gp, y_best)

        best_x = X_candidates[np.argmax(ei)]

        print(f"\n{func_name}")
        print(f"Best observed: {y_best:.6f}")
        print(f"Suggested query: {format_query(best_x)}")

if __name__ == "__main__":
    main()