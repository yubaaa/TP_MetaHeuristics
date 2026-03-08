from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import math
import matplotlib

matplotlib.use("Agg")  # ← fix GUI thread warnings
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

os.makedirs("static", exist_ok=True)


def serialize_trajectory(traj):
    """Convertit une liste de np.ndarray en liste de listes Python."""
    return [p.tolist() if isinstance(p, np.ndarray) else list(p) for p in traj]


# ======================
# BENCHMARK FUNCTIONS
# ======================


def f1(x):
    return sum(val**2 for val in x)


def f2(x):
    sum_part = sum(abs(val) for val in x)
    prod_part = 1
    for val in x:
        prod_part *= abs(val)
    return sum_part + prod_part


def f5(x):
    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (1 - x[i]) ** 2
    return result


def f8(x):
    return sum(-val * math.sin(math.sqrt(abs(val))) for val in x)


def f9(x):
    return sum(val**2 - 10 * math.cos(2 * math.pi * val) + 10 for val in x)


def f11(x):
    sum_part = sum(val**2 for val in x) / 4000
    prod_part = 1
    for i, val in enumerate(x):
        prod_part *= math.cos(val / math.sqrt(i + 1))
    return 1 + sum_part - prod_part


functions = {"F1": f1, "F2": f2, "F5": f5, "F8": f8, "F9": f9, "F11": f11}

domains = {
    "F1": (-100, 100),
    "F2": (-10, 10),
    "F5": (-30, 30),
    "F7": (-128, 128),
    "F8": (-500, 500),
    "F9": (-5.12, 5.12),
    "F11": (-600, 600),
}


# ======================
# SHARED PLOT HELPER
# ======================


def save_plots(
    func,
    D,
    lb,
    ub,
    all_positions,
    history,
    first_position,
    gbest,
    curve,
    prefix="",
    avg_curve=None,
    trajectory=None,
    stagnation_iter=None,
    snap_first=None,
):

    import numpy as np
    import matplotlib.pyplot as plt

    # ==============================
    # 1️ Convergence curve
    # ==============================

    plt.figure()
    plt.plot(curve, color="red", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title(f"{prefix} Convergence")
    plt.grid(True)

    if stagnation_iter is not None:
        plt.axvline(stagnation_iter, color="gray", linestyle="--", label="Stagnation")
        plt.legend()

    plt.tight_layout()
    plt.savefig("static/convergence.png", dpi=120)
    plt.close()

    # ==============================
    # 2️ Average fitness
    # ==============================

    if avg_curve is not None:

        plt.figure()
        plt.plot(avg_curve, color="blue", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Average fitness")
        plt.title("Average Fitness")
        plt.grid(True)

        if stagnation_iter is not None:
            plt.axvline(stagnation_iter, color="gray", linestyle="--")

        plt.tight_layout()
        plt.savefig("static/avg_fitness.png", dpi=120)
        plt.close()

    # ==============================
    # 3️ Trajectory
    # ==============================

    if trajectory is not None:

        traj = np.array(trajectory)

        plt.figure()

        if traj.ndim == 2 and traj.shape[1] >= 2:
            plt.plot(traj[:, 0], label="x1")
            plt.plot(traj[:, 1], label="x2")
            plt.legend()
        else:
            plt.plot(traj)

        plt.xlabel("Iteration")
        plt.ylabel("Position")
        plt.title("Trajectory of first solution")
        plt.grid(True)

        if stagnation_iter is not None:
            plt.axvline(stagnation_iter, color="gray", linestyle="--")

        plt.tight_layout()
        plt.savefig("static/trajectory.png", dpi=120)
        plt.close()

    # ==============================
    # 4 Search history (Contour)
    # ==============================

    if D < 2:
        return

    # grid pour contour
    resolution = 140
    x_range = np.linspace(lb, ub, resolution)
    y_range = np.linspace(lb, ub, resolution)

    X, Y = np.meshgrid(x_range, y_range)

    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):

            point = [X[i, j], Y[i, j]]

            if D > 2:
                point += [0] * (D - 2)

            Z[i, j] = func(point)

    # ==============================
    # récupérer toutes les positions explorées
    # ==============================

    all_points = []

    for iteration in all_positions:
        for p in iteration:
            all_points.append(p)

    all_points = np.array(all_points)

    # ==============================
    # Best position history
    # ==============================

    hx = [p[0] for p in history]
    hy = [p[1] for p in history]

    # ==============================
    # Plot
    # ==============================

    plt.figure(figsize=(7, 7))

    # contour surface
    plt.contourf(X, Y, Z, levels=40, cmap="YlGn", alpha=0.8)
    plt.contour(X, Y, Z, levels=40, colors="gray", linewidths=0.4)

    if len(all_points) > 0:
        plt.scatter(
            all_points[:, 0],
            all_points[:, 1],
            color="black",
            s=6,
            alpha=0.4,
            label="Population search",
        )

    plt.scatter(hx, hy, color="orange", s=40, label="Best per iteration")

    plt.plot(hx, hy, color="orange", linewidth=1.5)

    plt.scatter(gbest[0], gbest[1], color="red", s=200, marker="*", label="Global best")

    if stagnation_iter is not None and stagnation_iter < len(hx):

        plt.scatter(
            hx[stagnation_iter],
            hy[stagnation_iter],
            color="purple",
            s=160,
            marker="D",
            label="Stagnation",
        )

    plt.xlim(lb, ub)
    plt.ylim(lb, ub)

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title(f"Search History ({prefix})")

    plt.legend()

    plt.tight_layout()

    plt.savefig("static/search_history_contour.png", dpi=200)

    plt.close()


# ======================
# ROUTES API
# ======================


@app.route("/generate", methods=["POST"])
def generate_single():
    data = request.get_json()
    D = int(data["dimension"])
    min_val = float(data["min"])
    max_val = float(data["max"])
    vector = [random.uniform(min_val, max_val) for _ in range(D)]
    return jsonify(vector)


@app.route("/generate-multiple", methods=["POST"])
def generate_multiple():
    data = request.get_json()
    D = int(data["dimension"])
    min_val = float(data["min"])
    max_val = float(data["max"])
    size = int(data["size"])
    vectors = np.random.uniform(min_val, max_val, (size, D))
    return jsonify(vectors.tolist())


def generate_plots(vectors, function_name):
    func = functions[function_name]
    vectors = np.array(vectors)
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]
    if function_name == "F8":
        x_min, x_max, y_min, y_max = -500, 500, -500, 500
    elif function_name == "F11":
        x_min, x_max, y_min, y_max = -600, 600, -600, 600
    elif function_name == "F9":
        x_min, x_max, y_min, y_max = -5.12, 5.12, -5.12, 5.12
    elif function_name == "F2":
        x_min, x_max, y_min, y_max = -10, 10, -10, 10
    elif function_name == "F5":
        x_min, x_max, y_min, y_max = -30, 30, -30, 30
    elif function_name == "F7":
        x_min, x_max, y_min, y_max = -128, 128, -128, 128
    else:
        x_min, x_max, y_min, y_max = -100, 100, -100, 100
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])
    fitness = [func(v) for v in vectors]
    best_index = np.argmin(fitness)
    best_point = vectors[best_index]
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="viridis")
    ax1.set_title(f"Function ({function_name})")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(X, Y, Z, levels=20)
    ax2.scatter(x_vals, y_vals, color="black")
    ax2.scatter(best_point[0], best_point[1], color="red", s=100)
    ax2.set_title(f"Search History ({function_name})")
    plt.tight_layout()
    plt.savefig("static/plot.png")
    plt.close()
    return best_point.tolist()


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    vector = list(map(float, data["vector"]))
    function_name = data["function"]
    if function_name not in functions:
        return jsonify({"error": "Invalid function"}), 400
    return jsonify(functions[function_name](vector))


@app.route("/evaluate-csv", methods=["POST"])
def evaluate_csv():
    data = request.get_json()
    vectors = data.get("vectors", [])
    function_name = data.get("function")
    if function_name not in functions:
        return jsonify({"error": "Invalid function"}), 400
    if len(vectors) == 0:
        return jsonify({"error": "No vectors provided"}), 400
    vectors = [list(map(float, v)) for v in vectors]
    func = functions[function_name]
    results = [func(v) for v in vectors]
    best_index = results.index(min(results))
    worst_index = results.index(max(results))
    generate_plots(vectors, function_name)
    return jsonify(
        {
            "best_solution": vectors[best_index],
            "best_value": results[best_index],
            "worst_value": results[worst_index],
            "image_url": "http://localhost:5000/static/plot.png",
        }
    )


@app.route("/evaluate-multiple-runs", methods=["POST"])
def evaluate_multiple_runs():
    data = request.get_json()
    function_name = data.get("function")
    num_runs = int(data.get("runs", 10))
    pop_size = int(data.get("size", 30))
    D = int(data.get("dimension", 30))
    if function_name not in functions:
        return jsonify({"error": "Invalid function"}), 400
    func = functions[function_name]
    range_min, range_max = domains.get(function_name, (-100, 100))
    global_best = None
    global_worst = None
    sum_best = 0
    all_best_values = []
    all_vectors = []
    for r in range(num_runs):
        population = np.random.uniform(range_min, range_max, (pop_size, D))
        run_best = None
        run_worst = None
        for i in range(pop_size):
            val = func(population[i].tolist())
            if run_best is None or val < run_best:
                run_best = val
            if run_worst is None or val > run_worst:
                run_worst = val
        all_best_values.append(run_best)
        all_vectors.append(population)
        if global_best is None or run_best < global_best:
            global_best = run_best
        if global_worst is None or run_worst > global_worst:
            global_worst = run_worst
        sum_best += run_best
    avg = sum_best / num_runs
    std = float(np.std(all_best_values, ddof=1)) if num_runs > 1 else 0.0
    generate_plots(np.vstack(all_vectors), function_name)
    return jsonify(
        {
            "best": global_best,
            "worst": global_worst,
            "mean": avg,
            "std": std,
            "image_url": "http://localhost:5000/static/plot.png",
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# PSO
# ══════════════════════════════════════════════════════════════════════════════


def _pso_single_run(func, D, lb, ub, T, N, w, c1, c2):
    k = 0.2
    v_max = k * (ub - lb)
    P = np.random.uniform(lb, ub, (N, D))
    V = np.zeros((N, D))
    fitness = np.array([func(P[i]) for i in range(N)])
    best_idx = np.argmin(fitness)
    x_star = P[best_idx].copy()
    x_star_fit = fitness[best_idx]
    pbest = P.copy()
    pbest_fit = fitness.copy()
    snap_first = P.copy()
    first_position = x_star.copy()
    curve = [float(x_star_fit)]
    avg_curve = [float(np.mean(fitness))]
    all_positions = [P.copy()]
    history = [(float(x_star[0]), float(x_star[1]))] if D >= 2 else []
    trajectory = [P[0].copy()]
    stagnation_iter = None
    stagnation_count = 0
    prev_best = x_star.copy()
    prev_best_fit = x_star_fit  # On suit la fitness
    t = 0

    while True:
        for i in range(N):
            if not np.array_equal(x_star, P[i]):
                for j in range(D):
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    V[i, j] = (
                        w * V[i, j]
                        + c1 * r1 * (pbest[i, j] - P[i, j])
                        + c2 * r2 * (x_star[j] - P[i, j])
                    )
                    V[i, j] = np.clip(V[i, j], -v_max, v_max)
                    P[i, j] = np.clip(P[i, j] + V[i, j], lb, ub)
        fitness = np.array([func(P[i]) for i in range(N)])

        # Mise à jour du gbest
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < x_star_fit:
            x_star_fit = fitness[current_best_idx]
            x_star = P[current_best_idx].copy()
            stagnation_count = 0  # On a trouvé mieux, on reset
        else:
            stagnation_count += 1  # Pas d'amélioration

        # Enregistrement de la première stagnation
        if stagnation_count >= 30 and stagnation_iter is None:
            stagnation_iter = t

        for i in range(N):
            if x_star_fit > fitness[i]:
                x_star = P[i].copy()
                x_star_fit = fitness[i]
            if pbest_fit[i] > fitness[i]:
                pbest[i] = P[i].copy()
                pbest_fit[i] = fitness[i]
        t += 1

        if stagnation_count >= 50:  # Seuil d'arrêt anticipé
            break

        curve.append(float(x_star_fit))
        avg_curve.append(float(np.mean(fitness)))
        all_positions.append(P.copy())
        if D >= 2:
            history.append((float(x_star[0]), float(x_star[1])))
        trajectory.append(P[0].copy())

        if t >= T or stagnation_count >= 30:
            break

    return {
        "gbest": x_star,
        "gbest_fit": float(x_star_fit),
        "curve": curve,
        "history": history,
        "all_positions": all_positions,
        "first_position": first_position,
        "snap_first": snap_first,
        "avg_curve": avg_curve,
        "trajectory": serialize_trajectory(trajectory),
        "stagnation_iter": stagnation_iter,
    }


def pso_prof(func, D, lb, ub, T, N=30, w=0.3, c1=1.4, c2=1.4, R=1):
    if R <= 1:
        run = _pso_single_run(func, D, lb, ub, T, N, w, c1, c2)
        save_plots(
            func,
            D,
            lb,
            ub,
            run["all_positions"],
            run["history"],
            run["first_position"],
            run["gbest"],
            run["curve"],
            prefix="PSO",
            avg_curve=run["avg_curve"],
            trajectory=run["trajectory"],
            stagnation_iter=run["stagnation_iter"],
            snap_first=run["snap_first"],
        )
        stats = {
            "best": run["gbest_fit"],
            "worst": run["gbest_fit"],
            "mean": float(run["gbest_fit"]),  # Mean of 1 value is the value itself
            "std": 0.0,  # Std dev of 1 value is 0
            "all_best_fits": [run["gbest_fit"]],
        }
        return (
            run["gbest"].tolist(),
            run["gbest_fit"],
            run["curve"],
            run["history"],
            run["avg_curve"],
            run["trajectory"],
            run["stagnation_iter"],
            stats,
        )

    all_runs = []
    all_best_fits = []
    for _ in range(R):
        run = _pso_single_run(func, D, lb, ub, T, N, w, c1, c2)
        all_runs.append(run)
        all_best_fits.append(run["gbest_fit"])
    best_run_idx = int(np.argmin(all_best_fits))
    worst_run_idx = int(np.argmax(all_best_fits))
    best_run = all_runs[best_run_idx]
    save_plots(
        func,
        D,
        lb,
        ub,
        best_run["all_positions"],
        best_run["history"],
        best_run["first_position"],
        best_run["gbest"],
        best_run["curve"],
        prefix=f"PSO (best of {R} runs)",
        avg_curve=best_run["avg_curve"],
        trajectory=best_run["trajectory"],
        stagnation_iter=best_run["stagnation_iter"],
        snap_first=best_run["snap_first"],
    )
    stats = {
        "best": float(all_best_fits[best_run_idx]),
        "worst": float(all_best_fits[worst_run_idx]),
        "mean": float(np.mean(all_best_fits)),
        "std": float(np.std(all_best_fits, ddof=1)),
        "all_best_fits": all_best_fits,
    }
    return (
        best_run["gbest"].tolist(),
        best_run["gbest_fit"],
        best_run["curve"],
        best_run["history"],
        best_run["avg_curve"],
        best_run["trajectory"],
        best_run["stagnation_iter"],
        stats,
    )


@app.route("/evaluate-PSO", methods=["POST"])
def evaluate_PSO():
    data = request.get_json()
    c1 = float(data.get("c1", 1.4))
    c2 = float(data.get("c2", 1.4))
    w = float(data.get("w", 0.3))
    function_name = data.get("function")
    T = int(data.get("numIter", 200))
    N = int(data.get("size", 30))
    D = int(data.get("dimension", 2))
    R = int(data.get("runs", 1))
    if function_name not in functions:
        return jsonify({"error": "Invalid function"}), 400
    func = functions[function_name]
    range_min, range_max = domains.get(function_name, (-100, 100))
    (
        best_position,
        best_fitness,
        curve,
        history,
        avg_curve,
        trajectory,
        stagnation_iter,
        stats,
    ) = pso_prof(
        func=func, D=D, lb=range_min, ub=range_max, T=T, N=N, w=w, c1=c1, c2=c2, R=R
    )
    return jsonify(
        {
            "best_position": best_position,
            "best_fitness": best_fitness,
            "curve": curve,
            "image_url": "http://localhost:5000/static/convergence.png",
            "image_url2": "http://localhost:5000/static/search_history_contour.png",
            "image_url3": "http://localhost:5000/static/avg_fitness.png",
            "image_url4": "http://localhost:5000/static/trajectory.png",
            "image_url5": "http://localhost:5000/static/search_history_contour.png",
            "history": history,
            "avg_curve": avg_curve,
            "trajectory": trajectory,
            "stagnation_iter": stagnation_iter,
            "stats": {
                "best": stats["best"],
                "worst": stats["worst"],
                "mean": stats["mean"],
                "std": stats["std"],
                "all_best_fits": stats["all_best_fits"],
                "num_runs": R,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
