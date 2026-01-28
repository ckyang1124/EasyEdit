import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import argparse

# ==========================================
# 1. 數據準備
# ==========================================
methods = ["FT (LLM)", "FT (Audio)", "KE", "MEND", "UnKE", "I-IKE", "IE-IKE", "WISE"]
target_models = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]

# CSV Path
# Assuming the script is in paper_utils/plot_scripts and csv is in paper_utils/csvs
csv_path = os.path.join(os.path.dirname(__file__), "../csvs/single_editing_all.csv")
if not os.path.exists(csv_path):
    # Fallback to absolute path based on workspace
    csv_path = "/home/biao/data/research_work/lalmke/EasyEdit/model_outputs/paper_utils/csvs/single_editing_all.csv"

# Initialize data storage
# Dictionary: model -> 'gen'/'loc' -> numpy array (methods x types)
data_store = {
    m: {
        "gen": np.zeros((len(methods), 3)),  # 3 types for Generality
        "loc": np.zeros((len(methods), 4)),  # 4 types for Audio Locality
    }
    for m in target_models
}


def parse_float(val):
    try:
        val = val.strip()
        if not val:
            return 0.0
        return float(val)
    except ValueError:
        return 0.0


try:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

        # Skip headers (row 0 and 1)
        data_rows = rows[2:]

        current_model = None

        for row in data_rows:
            if not row or len(row) < 2:
                continue

            model_name = row[0].strip()
            if model_name:
                if model_name in target_models:
                    current_model = model_name

            if current_model not in target_models:
                continue

            method_name = row[1].strip()
            attr = row[2].strip()

            if attr == "ALL" and method_name in methods:
                method_idx = methods.index(method_name)

                # Careful with indices.
                # Row structure:
                # 0: Model, 1: Method, 2: Attr, 3: Rel, 4: Gen. Avg
                # 5: Gen. Type 1, 6: Gen. Type 2, 7: Gen. Type 3
                # 8: Audio Audio Loc. Avg
                # 9: Audio Loc. Type 1, 10: Audio Loc. Type 2, 11: Audio Loc. Type 3, 12: Audio Loc. Type 4

                # Gen. Types: columns 5, 6, 7
                if len(row) > 7:
                    data_store[current_model]["gen"][method_idx, 0] = parse_float(
                        row[5]
                    )
                    data_store[current_model]["gen"][method_idx, 1] = parse_float(
                        row[6]
                    )
                    data_store[current_model]["gen"][method_idx, 2] = parse_float(
                        row[7]
                    )

                # Audio Loc. Types: columns 9, 10, 11, 12
                if len(row) > 12:
                    data_store[current_model]["loc"][method_idx, 0] = parse_float(
                        row[9]
                    )
                    data_store[current_model]["loc"][method_idx, 1] = parse_float(
                        row[10]
                    )
                    data_store[current_model]["loc"][method_idx, 2] = parse_float(
                        row[11]
                    )
                    data_store[current_model]["loc"][method_idx, 3] = parse_float(
                        row[12]
                    )

except Exception as e:
    print(f"Error reading CSV: {e}")

# ==========================================
# 2. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(description="Generate radar charts for models.")
parser.add_argument(
    "--model",
    type=str,
    default="all",
    choices=target_models + ["all"],
    help="Model to plot (or 'all')",
)
parser.add_argument(
    "--show_legend",
    action="store_true",
    # default=True,
    # help="Show legend (default: True). Use --no_legend to hide.",
)
# parser.add_argument(
#     "--no_legend",
#     action="store_false",
#     dest="show_legend",
#     help="Do not show legend",
# )
args = parser.parse_args()

# ==========================================
# 3. 繪圖設定
# ==========================================
if args.model == "all":
    models_to_plot = target_models
    nrows = 3
    figsize = (16, 20)
else:
    models_to_plot = [args.model]
    nrows = 1
    figsize = (16, 7)

fig, axes = plt.subplots(nrows, 2, figsize=figsize, subplot_kw=dict(polar=True))
axes = axes.flatten()

# 角度設定
N = len(methods)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 顏色設定
colors_gen = ["#4e79a7", "#f28e2b", "#e15759"]
colors_loc = ["#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
linestyles = ["-", "--", "-.", ":"]


def plot_radar(ax, data, colors, title, type_name, num_types):
    # 軸設定
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods, size=18)

    # Y軸 (分數)
    ax.set_rlabel_position(0)
    plt.setp(ax.get_yticklabels(), fontsize=11, color="grey")
    ax.set_ylim(0, 105)

    # 畫線
    for i in range(num_types):
        values = data[:, i].tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle=linestyles[i % 4],
            label=f"Type {i+1} {type_name}",
            color=colors[i],
            marker="o",
            markersize=4,
        )
        ax.fill(angles, values, color=colors[i], alpha=0.05)

    ax.set_title(title, size=20, weight="bold", y=1.08)


# --- 繪製 Charts ---
for i, m_name in enumerate(models_to_plot):
    # Generality
    gen_data = data_store[m_name]["gen"]
    plot_radar(axes[i * 2], gen_data, colors_gen, f"Generality", "Gen.", 3)

    # Locality
    loc_data = data_store[m_name]["loc"]
    plot_radar(
        axes[i * 2 + 1],
        loc_data,
        colors_loc,
        f"Audio Locality",
        "Audio Loc.",
        4,
    )


if args.show_legend:
    # 取得 Legend Handles & Labels
    # Use axes[0] and axes[1] because we want the first row anyway
    lines_gen, labels_gen = axes[0].get_legend_handles_labels()
    lines_loc, labels_loc = axes[1].get_legend_handles_labels()

    # 分別放置 Legend 於最上方
    # Generality 左邊 (對應左欄)
    axes[0].legend(
        lines_gen,
        labels_gen,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=17,
        frameon=False,
    )
    # Locality 右邊 (對應右欄)
    axes[1].legend(
        lines_loc,
        labels_loc,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=17,
        frameon=False,
    )


plt.tight_layout()
# 調整間距，避免標題與下方圖表重疊
if nrows == 1:
    plt.subplots_adjust(wspace=-0.2, top=0.85)
else:
    plt.subplots_adjust(wspace=-0.6, hspace=0.3, top=0.91)

output_filename = (
    "radar_comparison.png"
    if args.model == "all"
    else f"radar_{args.model.replace(' ', '_')}.png"
)
output_path = output_filename
plt.savefig(output_path, bbox_inches="tight")
print(f"Radar chart saved to {os.path.abspath(output_path)}")
# plt.show()
