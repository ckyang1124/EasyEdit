import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
import sys

# Set up plot style
plt.rcParams.update({"font.size": 18})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sequence line chart from CSV."
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
        "--show_std", action="store_true", help="Show standard deviation shading"
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        default=True,
        help="Show legend (default: True). Use --no_legend to hide.",
    )
    parser.add_argument(
        "--no_legend",
        action="store_false",
        dest="show_legend",
        help="Do not show legend",
    )
    parser.add_argument(
        "--grid_layout",
        action="store_true",
        help="Use 3x2 grid layout (2 plots per row, 3 rows) instead of single row.",
    )
    return parser.parse_args()


def parse_val(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan, np.nan
    # Format "Mean (Std)"
    match = re.match(r"([0-9\.]+) \(([0-9\.]+)\)", str(s))
    if match:
        return float(match.group(1)), float(match.group(2))
    try:
        return float(s), 0.0
    except:
        return np.nan, np.nan


# Parse arguments
args = parse_args()
csv_path = args.csv_path

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    # Fallback to absolute path if running from different cwd
    fallback_path = "/home/biao/data/research_work/lalmke/EasyEdit/model_outputs/paper_utils/csvs/seqential_editing_all.csv"
    if os.path.exists(fallback_path):
        print(f"Using fallback path: {fallback_path}")
        csv_path = fallback_path
    else:
        exit()

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

# Data store: { Model: { inputs: { Metric: { Method: data } }, methods: [] } }
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

    # Initialize model data structure if new
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

    # Check gap
    gap_val = row[2]
    if pd.isna(gap_val):
        continue
    try:
        gap = int(gap_val)
    except ValueError:
        continue

    if current_method is None:
        continue

    # Store data for current model
    curr_plot_data = full_data[current_model]["plot_data"]

    for col_idx, metric_name in metrics_map.items():
        if col_idx >= len(row):
            continue

        val_str = row[col_idx]
        mean, std = parse_val(val_str)

        if current_method not in curr_plot_data[metric_name]:
            curr_plot_data[metric_name][current_method] = {
                "gaps": [],
                "means": [],
                "stds": [],
            }

        curr_plot_data[metric_name][current_method]["gaps"].append(gap)
        curr_plot_data[metric_name][current_method]["means"].append(mean)
        curr_plot_data[metric_name][current_method]["stds"].append(std)


# Determine which models to plot
models_to_plot = []
if args.model_name.lower() == "all":
    # Try to find target models in the keys
    for tm in target_models:
        for k in full_data.keys():
            if tm.lower() in k.lower():
                models_to_plot.append(k)
                break
else:
    # Find specific model
    query = args.model_name.lower()
    for k in full_data.keys():
        if query in k.lower():
            models_to_plot.append(k)


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
    for m in methods:
        has_data = False
        for metric in metrics_map.values():
            if m in plot_data[metric]:
                means = plot_data[metric][m]["means"]
                if any(not np.isnan(x) for x in means):
                    has_data = True
                    break
        if has_data:
            valid_methods.append(m)

    # Setup subplots
    if args.grid_layout:
        # 3 rows, 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharey=True)
        axes_flat = axes.flatten()
        # Hide extra subplots if any (we have 5 metrics, grid has 6 slots)
        for i in range(len(metrics_map), len(axes_flat)):
            axes_flat[i].set_axis_off()
    else:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=True)
        axes_flat = axes.flatten()

    # Markers and colors
    markers = ["o", "v", "s", "p", "*", "h", "D", "d", "^", "<", ">"]
    # Use a distinct color map
    colors = plt.cm.tab10.colors

    lines = []  # For legend

    for idx, (col_idx, metric_name) in enumerate(metrics_map.items()):
        ax = axes_flat[idx]
        ax.set_title(metric_name, fontsize=24)
        ax.grid(True, linestyle="-", alpha=0.3, color="lightgray")

        for i, method in enumerate(valid_methods):
            if method not in plot_data[metric_name]:
                continue

            data = plot_data[metric_name][method]
            gaps = np.array(data["gaps"])
            means = np.array(data["means"])
            stds = np.array(data["stds"])

            # Sort
            sort_idx = np.argsort(gaps)
            gaps = gaps[sort_idx]
            means = means[sort_idx]
            stds = stds[sort_idx]

            # Filter nans
            mask = ~np.isnan(means)
            gaps = gaps[mask]
            means = means[mask]
            stds = stds[mask]

            if len(gaps) == 0:
                continue

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            (line,) = ax.plot(
                gaps,
                means,
                marker=marker,
                label=method,
                color=color,
                linewidth=2,
                markersize=6,
                fillstyle="none",
                markeredgewidth=1.5,
            )
            if args.show_std:
                ax.fill_between(
                    gaps,
                    means - stds,
                    means + stds,
                    color=color,
                    alpha=0.15,
                    edgecolor=None,
                )

            if idx == 0:
                lines.append(line)

        ax.set_xlabel("Edit gap", fontsize=20)
        ax.set_xticks([0, 1, 2, 3, 4, 5])

        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=20)

        ax.set_ylim(0, 100)

    # Create a single legend
    if args.show_legend:
        bbox_anchor = (0.5, 1.05) if args.grid_layout else (0.5, 1.1)
        fig.legend(
            lines,
            valid_methods,
            loc="upper center",
            bbox_to_anchor=bbox_anchor,
            ncol=len(valid_methods) if not args.grid_layout else 3,
            frameon=False,
            fontsize=22,
        )

    plt.tight_layout()
    output_filename = f"seq_line_{m_name.replace(' ', '_')}.png"
    output_path = output_filename
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Chart saved to {output_path}")
