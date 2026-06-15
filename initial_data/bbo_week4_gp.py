import os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

BASE_DIR = "."
FUNCTION_FOLDERS = [f"function_{i}" for i in range(1, 9)]
np.random.seed(42)

# -----------------------------
# ROUND 1
# -----------------------------
round1_inputs = [
    np.array([0.500000, 0.500000]),
    np.array([0.786108, 0.766213]),
    np.array([0.509324, 0.672224, 0.305324]),
    np.array([0.250628, 0.593730, 0.622854, 0.050000]),
    np.array([0.242292, 0.890450, 0.872711, 0.880542]),
    np.array([0.683147, 0.544891, 0.407203, 0.817682, 0.076647]),
    np.array([0.241025, 0.683264, 0.609997, 0.833195, 0.173365, 0.391061]),
    np.array([0.213526, 0.434710, 0.283977, 0.510215, 0.869933, 0.671704, 0.264676, 0.265407]),
]

round1_outputs = np.array([
    2.6752879910742468e-09,
    0.03838400221215603,
    -0.09560907240309474,
    -11.923018749116093,
    1275.2685405563973,
    -0.7363719843000791,
    0.1660141105990867,
    9.6407935213586
])

# -----------------------------
# ROUND 2
# -----------------------------
round2_inputs = [
    np.array([0.500000, 0.500000]),
    np.array([0.698233, 0.926189]),
    np.array([0.299972, 0.056166, 0.474967]),
    np.array([0.433973, 0.372814, 0.480541, 0.364847]),
    np.array([0.520726, 0.923750, 0.957539, 0.995516]),
    np.array([0.409642, 0.418040, 0.312205, 0.818542, 0.103561]),
    np.array([0.093431, 0.462969, 0.286137, 0.068213, 0.353520, 0.633206]),
    np.array([0.122085, 0.080745, 0.011016, 0.224789, 0.742507, 0.480219, 0.165680, 0.248194]),
]

round2_outputs = np.array([
    2.6752879910742468e-09,
    0.5285960143922023,
    -0.08700196832088761,
    -0.8835934440569733,
    3249.3430704956318,
    -0.46412435211066894,
    1.3016764549502793,
    9.9293867977869
])

# -----------------------------
# ROUND 3
# -----------------------------
round3_inputs = [
    np.array([0.374540, 0.950714]),
    np.array([0.711980, 0.834464]),
    np.array([0.800000, 0.750000, 0.400000]),
    np.array([0.402044, 0.411799, 0.423589, 0.426184]),
    np.array([0.518037, 0.938867, 0.985634, 0.990000]),
    np.array([0.478267, 0.145180, 0.613760, 0.951012, 0.074658]),
    np.array([0.021122, 0.455751, 0.197504, 0.153114, 0.360924, 0.773482]),
    np.array([0.262563, 0.065531, 0.161972, 0.074335, 0.950000, 0.465611, 0.139469, 0.226796]),
]

round3_outputs = np.array([
    -1.560646704467778e-117,
    0.5086800893414688,
    -0.012630818813151479,
    0.6285556396489267,
    3643.287564441108,
    -0.48783256521310503,
    1.325428742931525,
    9.8975309817194
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

def get_local_noise(func_idx):
    if func_idx == 4:   # Function 5
        return 0.008
    if func_idx in [6, 7]:  # Functions 7 and 8
        return 0.025
    if func_idx in [1, 2, 5]:  # Functions 2, 3, 6
        return 0.040
    return 0.060

def get_xi(func_idx):
    if func_idx == 0:   # Function 1
        return 0.120
    if func_idx == 3:   # Function 4
        return 0.040
    if func_idx in [1, 5]:
        return 0.020
    return 0.010

def propose_query_gp(X, y, func_idx, n_candidates=25000):
    d = X.shape[1]

    kernel = (
        ConstantKernel(1.0, (1e-5, 1e3))
        * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 15.0), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )
    gp.fit(X, y)

    if func_idx == 0:      # Function 1: explore hard
        n_global = 22000
    elif func_idx == 4:    # Function 5: exploit hard
        n_global = 3000
    elif func_idx == 3:    # Function 4: mixed, but less global than F1
        n_global = 12000
    else:
        n_global = 14000

    n_local = n_candidates - n_global
    global_candidates = np.random.uniform(0, 1, size=(n_global, d))

    y_order = np.argsort(y)
    top_k = min(5, len(y))
    top_points = X[y_order[-top_k:]]

    local_noise = get_local_noise(func_idx)
    per_top = max(1, n_local // top_k)

    local_candidates = []
    for pt in top_points:
        cand = pt + np.random.normal(0, local_noise, size=(per_top, d))
        cand = np.clip(cand, 0, 1)
        local_candidates.append(cand)

    local_candidates = np.vstack(local_candidates)
    X_candidates = np.vstack([global_candidates, local_candidates])

    y_best = np.max(y)
    xi = get_xi(func_idx)
    ei = expected_improvement(X_candidates, gp, y_best, xi=xi)

    best_idx = np.argmax(ei)
    return X_candidates[best_idx], gp

def main():
    print("\nWeek 4 GP Query Suggestions")
    print("-" * 50)

    lines = []

    for i, func_name in enumerate(FUNCTION_FOLDERS):
        folder_path = os.path.join(BASE_DIR, func_name)

        X_old = np.load(os.path.join(folder_path, "initial_inputs.npy"))
        y_old = np.load(os.path.join(folder_path, "initial_outputs.npy"))

        X = np.vstack([X_old, round1_inputs[i], round2_inputs[i], round3_inputs[i]])
        y = np.append(y_old, [round1_outputs[i], round2_outputs[i], round3_outputs[i]])

        query, gp = propose_query_gp(X, y, i)

        print(f"\n{func_name}")
        print(f"Shape X: {X.shape}, Shape y: {y.shape}")
        print(f"Best observed y: {np.max(y):.6f}")
        print(f"Suggested query: {format_query(query)}")

        lines.append(f"{func_name}: {format_query(query)}")

    with open("week4_queries.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print("\nSaved to week4_queries.txt")

if __name__ == "__main__":
    main()