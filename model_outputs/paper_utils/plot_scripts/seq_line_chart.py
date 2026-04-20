import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
import sys
from matplotlib.ticker import AutoMinorLocator

# Set up plot style

PAPER_FONT_SANS = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Nimbus Sans",
    "Liberation Sans",
    "DejaVu Sans",
]


plt.rcParams.update(
    {
        "font.size": 18,
        "font.family": "sans-serif",
        "font.sans-serif": PAPER_FONT_SANS,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#A9B4C2",
        "axes.linewidth": 1.0,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
    }
)

X_TICKS = [0, 1, 2, 3, 4, 5]
Y_TICKS = [0, 20, 40, 60, 80, 100]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "<", ">", "d", "*"]

COMBINED_SUBPLOT_WIDTH = 3.5
COMBINED_SUBPLOT_HEIGHT = 3.8
SINGLE_ROW_FIGSIZE = (19, 5.8)
GRID_FIGSIZE = (10.8, 13.2)
COMBINED_ACCURACY_LABEL_SIZE = 11
COMBINED_MODEL_NAME_SIZE = 19
COMBINED_ACCURACY_LABEL_PAD = 2
COMBINED_MODEL_NAME_X_OFFSET = -0.23
COMBINED_LEGEND_Y = 0.985
COMBINED_TOP_RECT_WITH_LEGEND = 0.95
STANDARD_GRID_LEGEND_Y = 1.0
STANDARD_ROW_LEGEND_Y = 1.01
STANDARD_TOP_RECT_WITH_LEGEND = 0.94


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


def get_valid_methods(plot_data, methods, metric_names):
    valid_methods = []
    for m in methods:
        has_data = False
        for metric in metric_names:
            if m in plot_data[metric]:
                means = plot_data[metric][m]["means"]
                if any(not np.isnan(x) for x in means):
                    has_data = True
                    break
        if has_data:
            valid_methods.append(m)
    return valid_methods


def prepare_series(method_data):
    gaps = np.array(method_data["gaps"])
    means = np.array(method_data["means"])
    stds = np.array(method_data["stds"])

    sort_idx = np.argsort(gaps)
    gaps = gaps[sort_idx]
    means = means[sort_idx]
    stds = stds[sort_idx]

    mask = ~np.isnan(means)
    gaps = gaps[mask]
    means = means[mask]
    stds = stds[mask]
    return gaps, means, stds


def build_method_style_map(methods):
    palette = plt.get_cmap("tab20").colors
    style_map = {}
    for idx, method in enumerate(methods):
        style_map[method] = {
            "color": palette[(idx + 1) % len(palette)],
            "marker": MARKERS[idx % len(MARKERS)],
            "linestyle": LINE_STYLES[(idx // len(palette)) % len(LINE_STYLES)],
        }
    return style_map


def style_axis(
    ax,
    title=None,
    title_size=18,
    xlabel=None,
    ylabel=None,
    hide_left=False,
    hide_bottom=False,
    label_size=13,
    tick_size=11,
):
    ax.set_facecolor("#FBFCFE")
    ax.set_axisbelow(True)
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(0, 103)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(
        True, which="major", linestyle="-", linewidth=0.9, alpha=0.28, color="#8FA0B3"
    )
    ax.grid(
        True, which="minor", linestyle="-", linewidth=0.6, alpha=0.12, color="#8FA0B3"
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
        length=4.0,
        width=0.9,
        color="#5E6977",
    )
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.7, color="#7A8491")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A9B4C2")
    ax.spines["bottom"].set_color("#A9B4C2")

    if title:
        ax.set_title(title, fontsize=title_size, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=8)
    if hide_left:
        ax.tick_params(labelleft=False)
    if hide_bottom:
        ax.tick_params(labelbottom=False)


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

model_display_name_map = {
    "DeSTA2.5": "DeSTA",
    "Qwen2-Audio": "Qwen",
    "Audio Flamingo 3": "AF",
}


def get_display_model_name(model_name):
    lower_name = model_name.lower()
    for source_name, display_name in model_display_name_map.items():
        if source_name.lower() in lower_name:
            return display_name

    if "desta" in lower_name:
        return "DeSTA"
    if "qwen" in lower_name:
        return "Qwen"
    if "flamingo" in lower_name:
        return "AF"
    return model_name


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
model_query = args.model_name.lower()
models_to_plot = []
combined_mode = model_query == "combined"

if combined_mode or model_query == "all":
    # Try to find target models in the keys
    for tm in target_models:
        for k in full_data.keys():
            if tm.lower() in k.lower():
                models_to_plot.append(k)
                break
else:
    # Find specific model
    for k in full_data.keys():
        if model_query in k.lower():
            models_to_plot.append(k)


if not models_to_plot:
    print(
        f"No matching models found for '{args.model_name}'. Available: {list(full_data.keys())}"
    )
    exit()

print(f"Generating charts for: {models_to_plot}")

metric_names = list(metrics_map.values())

# Combined figure for target models: rows are models, columns are metrics
if combined_mode:
    print("Processing combined chart...")

    n_rows = len(models_to_plot)
    n_cols = len(metric_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(COMBINED_SUBPLOT_WIDTH * n_cols, COMBINED_SUBPLOT_HEIGHT * n_rows),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    if n_rows == 1:
        axes = np.array([axes])

    valid_methods_by_model = {}
    method_order = []
    for m_name in models_to_plot:
        plot_data = full_data[m_name]["plot_data"]
        methods = full_data[m_name]["methods"]
        valid_methods = get_valid_methods(plot_data, methods, metric_names)
        valid_methods_by_model[m_name] = valid_methods
        for method in valid_methods:
            if method not in method_order:
                method_order.append(method)

    method_styles = build_method_style_map(method_order)
    legend_lines = {}

    for row_idx, m_name in enumerate(models_to_plot):
        print(f"Processing {m_name}...")
        display_model_name = get_display_model_name(m_name)
        plot_data = full_data[m_name]["plot_data"]
        valid_methods = valid_methods_by_model[m_name]

        for col_idx, metric_name in enumerate(metric_names):
            ax = axes[row_idx, col_idx]
            style_axis(
                ax,
                title=metric_name if row_idx == 0 else None,
                title_size=19,
                xlabel="Evaluation Offset" if row_idx == n_rows - 1 else None,
                ylabel="Accuracy (%)" if col_idx == 0 else None,
                hide_left=col_idx != 0,
                hide_bottom=row_idx != n_rows - 1,
                label_size=14,
                tick_size=12,
            )

            if col_idx == 0:
                ax.yaxis.label.set_size(COMBINED_ACCURACY_LABEL_SIZE)
                ax.yaxis.label.set_color("#4A5563")
                ax.yaxis.labelpad = COMBINED_ACCURACY_LABEL_PAD
                ax.text(
                    COMBINED_MODEL_NAME_X_OFFSET,
                    0.5,
                    display_model_name,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=COMBINED_MODEL_NAME_SIZE,
                    fontweight="semibold",
                    va="center",
                    ha="center",
                    color="#1F2D3A",
                )

            for i, method in enumerate(valid_methods[::-1]):
                if method not in plot_data[metric_name]:
                    continue

                data = plot_data[metric_name][method]
                gaps, means, stds = prepare_series(data)
                if len(gaps) == 0:
                    continue

                style = method_styles[method]

                (line,) = ax.plot(
                    gaps,
                    means,
                    marker=style["marker"],
                    label=method,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=3.0,
                    markersize=9.0,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    alpha=0.95,
                )

                if args.show_std:
                    ax.fill_between(
                        gaps,
                        means - stds,
                        means + stds,
                        color=style["color"],
                        alpha=0.10,
                        edgecolor=None,
                    )

                if method not in legend_lines:
                    legend_lines[method] = line

    if args.show_legend and legend_lines:
        legend_methods = method_order  # list(legend_lines.keys())
        legend_handles = [legend_lines[m] for m in legend_methods]
        legend = fig.legend(
            legend_handles,
            legend_methods,
            loc="upper center",
            bbox_to_anchor=(0.5, COMBINED_LEGEND_Y),
            ncol=len(legend_methods),
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            fontsize=13,
            handlelength=2.0,
            columnspacing=1.0,
            handletextpad=0.5,
            borderpad=0.4,
        )
        legend.get_frame().set_edgecolor("#CDD6E0")
        legend.get_frame().set_linewidth(0.8)

    top_rect = (
        COMBINED_TOP_RECT_WITH_LEGEND if args.show_legend and legend_lines else 0.97
    )
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    output_path = "seq_line_combined.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Chart saved to {output_path}")
    sys.exit(0)

# Generate charts for each model
for m_name in models_to_plot:
    print(f"Processing {m_name}...")
    plot_data = full_data[m_name]["plot_data"]
    methods = full_data[m_name]["methods"]

    # Filter out methods with no data
    valid_methods = get_valid_methods(plot_data, methods, metric_names)

    # Setup subplots
    if args.grid_layout:
        # 3 rows, 2 columns
        fig, axes = plt.subplots(3, 2, figsize=GRID_FIGSIZE, sharey=True)
        axes_flat = axes.flatten()
        # Hide extra subplots if any (we have 5 metrics, grid has 6 slots)
        for i in range(len(metrics_map), len(axes_flat)):
            axes_flat[i].set_axis_off()
    else:
        fig, axes = plt.subplots(1, 5, figsize=SINGLE_ROW_FIGSIZE, sharey=True)
        axes_flat = axes.flatten()
    fig.patch.set_facecolor("white")

    method_styles = build_method_style_map(valid_methods)

    lines = []  # For legend

    for idx, metric_name in enumerate(metric_names):
        ax = axes_flat[idx]

        if args.grid_layout:
            n_grid_cols = 2
            n_grid_rows = 3
            row_idx = idx // n_grid_cols
            col_idx = idx % n_grid_cols
            show_xlabel = row_idx == n_grid_rows - 1
            show_ylabel = col_idx == 0
        else:
            show_xlabel = idx == len(metric_names) // 2
            show_ylabel = idx == 0

        style_axis(
            ax,
            title=metric_name,
            title_size=22,
            xlabel="Evaluation Offset" if show_xlabel else None,
            ylabel="Accuracy (%)" if show_ylabel else None,
            hide_left=not show_ylabel,
            hide_bottom=False,
            label_size=18,
            tick_size=15,
        )

        for i, method in enumerate(valid_methods):
            if method not in plot_data[metric_name]:
                continue

            data = plot_data[metric_name][method]
            gaps, means, stds = prepare_series(data)

            if len(gaps) == 0:
                continue

            style = method_styles[method]

            (line,) = ax.plot(
                gaps,
                means,
                marker=style["marker"],
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.6,
                markersize=6.3,
                markerfacecolor="white",
                markeredgewidth=1.2,
                alpha=0.95,
            )
            if args.show_std:
                ax.fill_between(
                    gaps,
                    means - stds,
                    means + stds,
                    color=style["color"],
                    alpha=0.12,
                    edgecolor=None,
                )

            if idx == 0:
                lines.append(line)

    # Create a single legend
    if args.show_legend:
        bbox_anchor = (
            (0.5, STANDARD_GRID_LEGEND_Y)
            if args.grid_layout
            else (0.5, STANDARD_ROW_LEGEND_Y)
        )
        n_cols = (
            min(4, len(valid_methods))
            if args.grid_layout
            else min(6, len(valid_methods))
        )
        legend = fig.legend(
            lines,
            valid_methods,
            loc="upper center",
            bbox_to_anchor=bbox_anchor,
            ncol=n_cols,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            fontsize=16 if args.grid_layout else 18,
            handlelength=2.0,
            columnspacing=1.0,
            handletextpad=0.5,
            borderpad=0.4,
        )
        legend.get_frame().set_edgecolor("#CDD6E0")
        legend.get_frame().set_linewidth(0.8)

    top_rect = STANDARD_TOP_RECT_WITH_LEGEND if args.show_legend else 0.98
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    output_filename = f"seq_line_{m_name.replace(' ', '_')}.png"
    output_path = output_filename
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Chart saved to {output_path}")
