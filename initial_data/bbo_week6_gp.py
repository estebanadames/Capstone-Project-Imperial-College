import os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

BASE_DIR = "."
FUNCTION_FOLDERS = [f"function_{i}" for i in range(1, 9)]
np.random.seed(42)

# =========================================================
# LOAD INITIAL DATA
# =========================================================

def load_initial_data(func_name):
    X = np.load(os.path.join(BASE_DIR, func_name, "initial_inputs.npy"))
    y = np.load(os.path.join(BASE_DIR, func_name, "initial_outputs.npy"))
    return X, y

# =========================================================
# WEEK DATA
# =========================================================

week_inputs = [
    [
        np.array([0.500000, 0.500000]),
        np.array([0.500000, 0.500000]),
        np.array([0.374540, 0.950714]),
        np.array([0.820000, 0.180000]),
        np.array([0.850000, 0.150000]),
    ],

    [
        np.array([0.786108, 0.766213]),
        np.array([0.698233, 0.926189]),
        np.array([0.711980, 0.834464]),
        np.array([0.953645, 0.335520]),
        np.array([0.050000, 0.835151]),
    ],

    [
        np.array([0.509324, 0.672224, 0.305324]),
        np.array([0.299972, 0.056166, 0.474967]),
        np.array([0.800000, 0.750000, 0.400000]),
        np.array([0.308827, 0.265255, 0.050000]),
        np.array([0.950000, 0.880000, 0.650000]),
    ],

    [
        np.array([0.250628, 0.593730, 0.622854, 0.050000]),
        np.array([0.433973, 0.372814, 0.480541, 0.364847]),
        np.array([0.402044, 0.411799, 0.423589, 0.426184]),
        np.array([0.383288, 0.422349, 0.360947, 0.432722]),
        np.array([0.381991, 0.436932, 0.408895, 0.448280]),
    ],

    [
        np.array([0.242292, 0.890450, 0.872711, 0.880542]),
        np.array([0.520726, 0.923750, 0.957539, 0.995516]),
        np.array([0.518037, 0.938867, 0.985634, 0.990000]),
        np.array([0.619957, 0.998484, 0.998958, 0.350000]),
        np.array([0.528674, 0.947024, 0.990000, 0.995000]),
    ],

    [
        np.array([0.683147, 0.544891, 0.407203, 0.817682, 0.076647]),
        np.array([0.409642, 0.418040, 0.312205, 0.818542, 0.103561]),
        np.array([0.478267, 0.145180, 0.613760, 0.951012, 0.074658]),
        np.array([0.475533, 0.456239, 0.451066, 0.989182, 0.050000]),
        np.array([0.442663, 0.135005, 0.549385, 0.591010, 0.050000]),
    ],

    [
        np.array([0.241025, 0.683264, 0.609997, 0.833195, 0.173365, 0.391061]),
        np.array([0.093431, 0.462969, 0.286137, 0.068213, 0.353520, 0.633206]),
        np.array([0.021122, 0.455751, 0.197504, 0.153114, 0.360924, 0.773482]),
        np.array([0.020000, 0.433983, 0.203498, 0.123701, 0.405817, 0.723999]),
        np.array([0.041934, 0.480055, 0.255043, 0.220831, 0.366187, 0.697708]),
    ],

    [
        np.array([0.213526, 0.434710, 0.283977, 0.510215, 0.869933, 0.671704, 0.264676, 0.265407]),
        np.array([0.122085, 0.080745, 0.011016, 0.224789, 0.742507, 0.480219, 0.165680, 0.248194]),
        np.array([0.262563, 0.065531, 0.161972, 0.074335, 0.950000, 0.465611, 0.139469, 0.226796]),
        np.array([0.053688, 0.005909, 0.149388, 0.031281, 0.822629, 0.495575, 0.136192, 0.807897]),
        np.array([0.321159, 0.285100, 0.079277, 0.239005, 0.950000, 0.344383, 0.123569, 0.830702]),
    ],
]

week_outputs = [
    [2.6752879910742468e-09, 2.6752879910742468e-09, -1.560646704467778e-117, 1.2411689730836742e-151, 2.8556640528793443e-181],
    [0.03838400221215603, 0.5285960143922023, 0.5086800893414688, 0.08359001917456062, -0.014987991658531197],
    [-0.09560907240309474, -0.08700196832088761, -0.012630818813151479, -0.06816775168805957, -0.11790008081891142],
    [-11.923018749116093, -0.8835934440569733, 0.6285556396489267, 0.5312263478605925, -0.023165133710903785],
    [1275.2685405563973, 3249.3430704956318, 3643.287564441108, 1975.2659195490683, 3867.0650282755832],
    [-0.7363719843000791, -0.46412435211066894, -0.48783256521310503, -0.5991304321604517, -0.589937785370754],
    [0.1660141105990867, 1.3016764549502793, 1.325428742931525, 1.223668225491426, 1.6644973432756731],
    [9.6407935213586, 9.9293867977869, 9.8975309817194, 9.9469856420036, 9.8158126377346]
]

# =========================================================
# EI
# =========================================================

def expected_improvement(X, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-12)

    imp = mu - y_best - xi
    Z = imp / sigma

    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# =========================================================
# FORMAT
# =========================================================

def format_query(x):
    return "-".join([f"{v:.6f}" for v in x])

# =========================================================
# QUERY STRATEGY
# =========================================================

def propose_query(X, y, func_idx):

    d = X.shape[1]

    kernel = (
        ConstantKernel(1.0)
        * Matern(length_scale=np.ones(d), nu=2.5)
        + WhiteKernel(noise_level=1e-5)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42
    )

    gp.fit(X, y)

    # -----------------------------------------------------
    # Exploration vs exploitation tuning
    # -----------------------------------------------------

    if func_idx == 4:   # Function 5
        n_global = 3000
        local_noise = 0.015
        xi = 0.003

    elif func_idx == 6: # Function 7
        n_global = 5000
        local_noise = 0.020
        xi = 0.005

    elif func_idx == 7: # Function 8
        n_global = 8000
        local_noise = 0.030
        xi = 0.008

    elif func_idx == 0: # Function 1
        n_global = 28000
        local_noise = 0.080
        xi = 0.150

    else:
        n_global = 15000
        local_noise = 0.040
        xi = 0.020

    n_total = 30000
    n_local = n_total - n_global

    # -----------------------------------------------------
    # Global exploration
    # -----------------------------------------------------

    global_candidates = np.random.uniform(0, 1, size=(n_global, d))

    # -----------------------------------------------------
    # Local exploitation
    # -----------------------------------------------------

    top_k = min(5, len(y))

    top_idx = np.argsort(y)[-top_k:]
    top_points = X[top_idx]

    local_candidates = []

    per_top = max(1, n_local // top_k)

    for pt in top_points:
        cand = pt + np.random.normal(0, local_noise, size=(per_top, d))
        cand = np.clip(cand, 0.02, 0.98)
        local_candidates.append(cand)

    local_candidates = np.vstack(local_candidates)

    X_candidates = np.vstack([global_candidates, local_candidates])

    y_best = np.max(y)

    ei = expected_improvement(X_candidates, gp, y_best, xi)

    best_idx = np.argmax(ei)

    return X_candidates[best_idx]

# =========================================================
# MAIN
# =========================================================

def main():

    print("\nWeek 6 Query Suggestions")
    print("-" * 50)

    for i, func_name in enumerate(FUNCTION_FOLDERS):

        X_old, y_old = load_initial_data(func_name)

        X = np.vstack([X_old] + week_inputs[i])
        y = np.concatenate([y_old, week_outputs[i]])

        query = propose_query(X, y, i)

        print(f"\n{func_name}")
        print(f"Best observed output: {np.max(y):.6f}")
        print(f"Suggested query: {format_query(query)}")

if __name__ == "__main__":
    main()