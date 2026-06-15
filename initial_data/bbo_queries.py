import os
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = "initial_data"  # current folder
FUNCTION_FOLDERS = [f"function_{i}" for i in range(1, 9)]

# Noise scale per function
# More exploitation for simpler/unimodal functions
# More exploration for noisy/high-dimensional ones
NOISE_SCALES = {
    "function_1": 0.05,
    "function_2": 0.10,
    "function_3": 0.06,
    "function_4": 0.06,
    "function_5": 0.03,
    "function_6": 0.07,
    "function_7": 0.08,
    "function_8": 0.10,
}

np.random.seed(42)


def format_query(x):
    return "-".join(f"{xi:.6f}" for xi in x)


def generate_query(X, y, func_name):
    """
    Hybrid strategy:
    - Pick one of top-k best known points
    - Perturb locally
    - For higher/noisier functions, occasionally explore globally
    """
    d = X.shape[1]
    noise_scale = NOISE_SCALES.get(func_name, 0.05)

    # Top-k candidates
    k = min(3, len(y))
    top_k_idx = np.argsort(y)[-k:]
    chosen_idx = np.random.choice(top_k_idx)
    chosen_x = X[chosen_idx]

    # More global exploration for harder functions
    if func_name in ["function_7", "function_8"] and np.random.rand() < 0.35:
        new_x = np.random.uniform(0, 1, size=d)
    elif func_name == "function_2" and np.random.rand() < 0.25:
        new_x = np.random.uniform(0, 1, size=d)
    else:
        noise = np.random.normal(0, noise_scale, size=d)
        new_x = chosen_x + noise

    # Keep values in [0,1]
    new_x = np.clip(new_x, 0, 1)

    return new_x


def main():
    print("\nBBO Query Suggestions\n" + "-" * 40)

    for func_name in FUNCTION_FOLDERS:
        folder_path = os.path.join(BASE_DIR, func_name)

        inputs_path = os.path.join(folder_path, "initial_inputs.npy")
        outputs_path = os.path.join(folder_path, "initial_outputs.npy")

        if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
            print(f"{func_name}: missing files")
            continue

        X = np.load(inputs_path)
        y = np.load(outputs_path)

        print(f"\n{func_name}")
        print(f"Input shape:  {X.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Best observed output: {np.max(y):.6f}")

        query = generate_query(X, y, func_name)
        query_str = format_query(query)

        print(f"Suggested query: {query_str}")


if __name__ == "__main__":
    main()