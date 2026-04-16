import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse

# Set up plot style
plt.rcParams.update({"font.size": 16})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sequential-editing interference heatmaps from CSV."
    )
    parser.add_argument("model_name", type=str, help="Model name (e.g. DeSTA)")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../csvs/seqential_editing_all.csv"
        ),
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Write values on each heatmap cell.",
    )
    parser.add_argument(
        "--no_annotate",
        action="store_false",
        dest="annotate",
        help="Do not write values on each heatmap cell.",
    )
    parser.add_argument(
        "--show_colorbar",
        action="store_true",
        default=True,
        help="Show colorbar (default: True). Use --no_colorbar to hide.",
    )
    parser.add_argument(
        "--no_colorbar",
        action="store_false",
        dest="show_colorbar",
        help="Do not show colorbar.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap name. Low values become darker with 'magma'.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Minimum value for heatmap color scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=100.0,
        help="Maximum value for heatmap color scale.",
    )
    parser.add_argument(
        "--grid_layout",
        action="store_true",
        help="Use 3x2 grid layout (2 plots per row, 3 rows) instead of single row.",
    )
    parser.add_argument(
        "--hide_x_axis",
        action="store_true",
        help="Hide x ticks and x-axis label.",
    )
    parser.add_argument(
        "--hide_titles",
        action="store_true",
        help="Hide subplot titles.",
    )
    return parser.parse_args()


def parse_val(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan, np.nan
    # Format "Mean (Std)"
    match = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*\(([+-]?\d+(?:\.\d+)?)\)\s*$",
        str(s),
    )
    if match:
        return float(match.group(1)), float(match.group(2))
    try:
        return float(s), 0.0
    except ValueError:
        return np.nan, np.nan


def resolve_csv_path(initial_path):
    if os.path.exists(initial_path):
        return initial_path

    print(f"File not found: {initial_path}")
    fallback_paths = [
        "/home/biao/data/research_work/lalmke/EasyEdit/model_outputs/paper_utils/csvs/seqential_editing_all.csv",
    ]
    for candidate in fallback_paths:
        if os.path.exists(candidate):
            print(f"Using fallback path: {candidate}")
            return candidate

    return None


def collect_full_data(df, metrics_map):
    full_data = {}
    current_method = None
    current_model = None

    # Iterate rows starting from index 2 (skipping header rows)
    for i in range(2, len(df)):
        row = df.iloc[i]

        # Model column
        model_val = row[0]
        if not pd.isna(model_val) and str(model_val).strip() != "":
            current_model = model_val.strip()

        if current_model is None:
            continue

        if current_model not in full_data:
            full_data[current_model] = {
                "plot_data": {m: {} for m in metrics_map.values()},
                "methods": [],
            }

        # Method column
        method_val = row[1]
        if not pd.isna(method_val) and str(method_val).strip() != "":
            current_method = method_val.strip()
            if current_method not in full_data[current_model]["methods"]:
                full_data[current_model]["methods"].append(current_method)

        # Interference step column (originally named Gap)
        step_val = row[2]
        if pd.isna(step_val):
            continue
        try:
            interference_step = int(step_val)
        except ValueError:
            continue

        if current_method is None:
            continue

        curr_plot_data = full_data[current_model]["plot_data"]
        for col_idx, metric_name in metrics_map.items():
            if col_idx >= len(row):
                continue

            val_str = row[col_idx]
            mean, _ = parse_val(val_str)

            if current_method not in curr_plot_data[metric_name]:
                curr_plot_data[metric_name][current_method] = {
                    "steps": [],
                    "means": [],
                }

            curr_plot_data[metric_name][current_method]["steps"].append(
                interference_step
            )
            curr_plot_data[metric_name][current_method]["means"].append(mean)

    return full_data


def resolve_models_to_plot(full_data, model_name, target_models):
    models_to_plot = []
    if model_name.lower() == "all":
        for tm in target_models:
            for model_key in full_data.keys():
                if tm.lower() in model_key.lower():
                    models_to_plot.append(model_key)
                    break
    else:
        query = model_name.lower()
        for model_key in full_data.keys():
            if query in model_key.lower():
                models_to_plot.append(model_key)
    return models_to_plot


def get_interference_steps(plot_data, metric_names):
    steps = set()
    for metric_name in metric_names:
        for method_data in plot_data[metric_name].values():
            steps.update(method_data["steps"])
    return sorted(steps)


def build_metric_matrix(metric_data, methods, interference_steps):
    matrix = np.full((len(methods), len(interference_steps)), np.nan)
    step_to_idx = {step: idx for idx, step in enumerate(interference_steps)}

    for row_idx, method in enumerate(methods):
        if method not in metric_data:
            continue

        method_series = metric_data[method]
        step_to_values = {}
        for step, val in zip(method_series["steps"], method_series["means"]):
            if np.isnan(val):
                continue
            step_to_values.setdefault(step, []).append(val)

        for step, values in step_to_values.items():
            col_idx = step_to_idx.get(step)
            if col_idx is None:
                continue
            matrix[row_idx, col_idx] = float(np.nanmean(values))

    return matrix


def get_colormap(cmap_name):
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Invalid cmap '{cmap_name}', fallback to 'magma'.")
        cmap = plt.get_cmap("magma")

    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color="#d9d9d9")
    return cmap


# Parse arguments
args = parse_args()
csv_path = resolve_csv_path(args.csv_path)
if csv_path is None:
    raise SystemExit(1)

print(f"Reading data from {csv_path}...")
df = pd.read_csv(csv_path, header=None)

# Mappings based on column index
# 0: Model, 1: Method, 2: Gap
# 3: Rel.
# 4: Gen. (Avg)
# 8: Audio Loc. (Avg)
# 13: Text Loc.
# 14: Port.

metrics_map = {
    3: "Reliability",
    4: "Generality",
    8: "Audio Locality",
    13: "Text Locality",
    14: "Portability",
}

target_models = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]

full_data = collect_full_data(df, metrics_map)
models_to_plot = resolve_models_to_plot(full_data, args.model_name, target_models)


if not models_to_plot:
    print(
        f"No matching models found for '{args.model_name}'. Available: {list(full_data.keys())}"
    )
    exit()

print(f"Generating charts for: {models_to_plot}")

# Generate charts for each model
for m_name in models_to_plot:
    print(f"Processing {m_name}...")
    plot_data = full_data[m_name]["plot_data"]
    methods = full_data[m_name]["methods"]

    # Filter out methods with no data
    valid_methods = []
    for method in methods:
        has_data = False
        for metric in metrics_map.values():
            if method in plot_data[metric]:
                means = plot_data[metric][method]["means"]
                if any(not np.isnan(x) for x in means):
                    has_data = True
                    break
        if has_data:
            valid_methods.append(method)

    if not valid_methods:
        print(f"No valid data for model: {m_name}")
        continue

    interference_steps = get_interference_steps(plot_data, metrics_map.values())
    if not interference_steps:
        print(f"No interference steps found for model: {m_name}")
        continue

    # Setup subplots
    if args.grid_layout:
        # 3 rows, 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(16, 10.5))
        fig.subplots_adjust(wspace=0.16, hspace=0.28)
        axes_flat = axes.flatten()
        # Hide extra subplots if any (we have 5 metrics, grid has 6 slots)
        for i in range(len(metrics_map), len(axes_flat)):
            axes_flat[i].set_axis_off()
    else:
        fig, axes = plt.subplots(1, len(metrics_map), figsize=(30, 5.4))
        fig.subplots_adjust(left=0.06, right=0.92, wspace=0.04)
        axes_flat = np.atleast_1d(axes).flatten()

    cmap = get_colormap(args.cmap)
    heatmap_image = None

    for idx, (_, metric_name) in enumerate(metrics_map.items()):
        ax = axes_flat[idx]
        metric_matrix = build_metric_matrix(
            plot_data[metric_name], valid_methods, interference_steps
        )

        heatmap_image = ax.imshow(
            metric_matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        if args.hide_titles:
            ax.set_title("")
        else:
            ax.set_title(metric_name, fontsize=20)

        if args.hide_x_axis:
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Evaluation Offset", fontsize=16)
            ax.set_xticks(np.arange(len(interference_steps)))
            ax.set_xticklabels(interference_steps)
            ax.tick_params(axis="x", labelsize=12)

        ax.set_yticks(np.arange(len(valid_methods)))
        if idx == 0:
            ax.set_yticklabels(valid_methods)
            ax.tick_params(axis="y", labelsize=15)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Draw cell boundaries to make degradation transitions easier to compare.
        ax.set_xticks(np.arange(-0.5, len(interference_steps), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(valid_methods), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        if args.annotate:
            threshold = (args.vmin + args.vmax) / 2.0
            for row_idx in range(metric_matrix.shape[0]):
                for col_idx in range(metric_matrix.shape[1]):
                    val = metric_matrix[row_idx, col_idx]
                    if np.isnan(val):
                        continue
                    text_color = "white" if val < threshold else "black"
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=11,
                    )

    if args.show_colorbar and heatmap_image is not None:
        cbar = fig.colorbar(
            heatmap_image,
            ax=axes_flat[: len(metrics_map)],
            fraction=0.025,
            pad=0.01,
        )
        cbar.set_label("Metric Score (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=11)

    output_filename = f"seq_interference_heatmap_{m_name.replace(' ', '_')}.png"
    output_path = output_filename
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Chart saved to {output_path}")
    plt.close(fig)
