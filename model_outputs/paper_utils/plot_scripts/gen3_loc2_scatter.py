import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr

MODEL_ORDER = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]
METHOD_ORDER = ["LMT", "KE", "MEND", "UnKE", "WISE", "I-IKE", "IE-IKE", "PCT"]
MODEL_DISPLAY_NAME = {
    "DeSTA2.5": "DeSTA",
    "Qwen2-Audio": "Qwen",
    "Audio Flamingo 3": "AF",
}

DOT_SIZE_DEFAULT = 200
MODEL_COLOR = {
    "DeSTA2.5": "#2A9D8F",
    "Qwen2-Audio": "#E76F51",
    "Audio Flamingo 3": "#7B6DCC",
}

UNIFORM_BORDER_LINEWIDTH = 1.2
PAPER_FONT_SANS = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Nimbus Sans",
    "Liberation Sans",
    "DejaVu Sans",
]

LEGEND_HANDLETEXTPAD = 0.2
LEGEND_FONTSIZE = 10
LEGEND_TITLE_FONTSIZE = 11
LEGEND_COLUMNSPACING = 0.5
LEGEND_METHOD_COLUMNSPACING = 0.4
LEGEND_COMMON_Y = 0.80
LEGEND_X_START = 0.1


def ordered_unique(values, preferred_order):
    existing = set(values)
    ordered = [name for name in preferred_order if name in existing]
    extras = sorted(existing.difference(ordered))
    return ordered + extras


def main():
    # Get absolute path to the script and CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../csvs/single_editing_all.csv")

    # Read the CSV file
    df = pd.read_csv(csv_path, header=None, skiprows=2)

    # Define columns based on the CSV structure
    df.columns = [
        "Model",
        "Method",
        "Attr",
        "Rel",
        "Gen_Avg",
        "Gen_Type1",
        "Gen_Type2",
        "Gen_Type3",
        "Audio_Loc_Avg",
        "Audio_Loc_Type1",
        "Audio_Loc_Type2",
        "Audio_Loc_Type3",
        "Audio_Loc_Type4",
        "Text_Loc",
        "Port",
    ]

    df["Model"] = df["Model"].ffill()
    df["Method"] = df["Method"].ffill()

    df_all = df[df["Attr"] == "ALL"].copy()

    # Normalize Method names
    df_all["Method"] = df_all["Method"].replace(
        {"FT (LLM)": "LMT", "FT (Audio)": "PCT"}
    )

    df_all["Gen_Type3"] = pd.to_numeric(
        df_all["Gen_Type3"].astype(str).str.replace("%", ""), errors="coerce"
    )
    df_all["Audio_Loc_Type2"] = pd.to_numeric(
        df_all["Audio_Loc_Type2"].astype(str).str.replace("%", ""), errors="coerce"
    )

    df_all = df_all.dropna(subset=["Gen_Type3", "Audio_Loc_Type2"])

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "font.sans-serif": PAPER_FONT_SANS,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    models = ordered_unique(df_all["Model"].tolist(), MODEL_ORDER)
    methods = ordered_unique(df_all["Method"].tolist(), METHOD_ORDER)

    marker_values = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fallback_colors = plt.get_cmap("Set2")(np.linspace(0.1, 0.9, len(models)))
    model_to_color = {}
    for idx, m in enumerate(models):
        if m in MODEL_COLOR:
            model_to_color[m] = MODEL_COLOR[m]
        else:
            model_to_color[m] = fallback_colors[idx % len(fallback_colors)]

    method_to_marker = {
        m: marker_values[idx % len(marker_values)] for idx, m in enumerate(methods)
    }

    fig, axis = plt.subplots(figsize=(7, 7))

    for _, row in df_all.iterrows():
        axis.scatter(
            row["Gen_Type3"],
            row["Audio_Loc_Type2"],
            color=model_to_color[row["Model"]],
            marker=method_to_marker[row["Method"]],
            s=DOT_SIZE_DEFAULT,
            edgecolors="black",
            linewidths=UNIFORM_BORDER_LINEWIDTH,
            alpha=0.95,
        )

    # Regression Line
    x = df_all["Gen_Type3"]
    y = df_all["Audio_Loc_Type2"]
    m, b = np.polyfit(x, y, 1)

    sign_b = "+" if b >= 0 else "-"
    eq_text = rf"$y = {m:.4f}x {sign_b} {abs(b):.4f}$"

    axis.plot(
        x,
        m * x + b,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Trend",
    )

    # Add the equation text inside the plot
    axis.text(
        0.1,
        0.75,
        eq_text,
        transform=axis.transAxes,
        fontsize=14,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3", alpha=0.9, facecolor="white", edgecolor="gray"
        ),
    )

    axis.set_xlabel("TA-Gen. (%)", labelpad=8, fontsize=14)
    axis.set_ylabel("Intra-Loc. (%)", labelpad=8, fontsize=14)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", labelsize=14)
    axis.tick_params(axis="y", labelsize=14)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    model_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=9,
            markerfacecolor=model_to_color[m],
            markeredgecolor="black",
            label=MODEL_DISPLAY_NAME.get(m, m),
        )
        for m in models
    ]

    method_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=method_to_marker[m],
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="black",
            label=m,
        )
        for m in methods
    ]

    # legend_trend = fig.legend(
    #     title="Trend",
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 0.9),
    #     frameon=True,
    # )

    fig.legend(
        handles=model_handles,
        title="Model (Color)",
        loc="lower left",
        bbox_to_anchor=(LEGEND_X_START, LEGEND_COMMON_Y),
        frameon=True,
        ncol=2,
        columnspacing=LEGEND_COLUMNSPACING,
        handletextpad=LEGEND_HANDLETEXTPAD,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )

    fig.legend(
        handles=method_handles,
        title="Method (Marker)",
        loc="lower left",
        bbox_to_anchor=(LEGEND_X_START + 0.3, LEGEND_COMMON_Y),
        ncol=4,
        columnspacing=LEGEND_METHOD_COLUMNSPACING,
        handletextpad=LEGEND_HANDLETEXTPAD,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.12)

    # Save the plot in the same directory as the script
    output_path = os.path.join(script_dir, "gen3_loc2_scatter.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    print(f"Plot saved successfully to {output_path}")


if __name__ == "__main__":
    main()
