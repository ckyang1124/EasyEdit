import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Data ──────────────────────────────────────────────────────────────────────
models = ["DeSTA", "Qwen", "AF"]
methods = ["LMT", "KE", "MEND", "UnKE", "WISE", "IE-IKE", "I-IKE", "PCT"]

raw = [
    [0.30, 47.77, 44.25, 2.23, 0.14, 0.00, 0.78, 1.35],
    [0.34, 11.97, 0.85, 2.30, 0.20, 1.28, 0.34, 0.20],
    [0.00, 9.67, 0.00, 6.02, 0.00, 0.00, 0.00, 0.00],
]
matrix = np.array(raw)
bold_mask = matrix == matrix.max(axis=1, keepdims=True)

# ── Typography ────────────────────────────────────────────────────────────────
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
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": PAPER_FONT_SANS,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

# ── Colormap ──────────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "paper",
    ["#FAFAFA", "#FDE8D8", "#E07B54", "#9C2A10", "#5C0A00"],
    N=512,
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.0, 2.0))

vmax = matrix.max()
im = ax.imshow(
    matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest"
)

# ── Cell separators ───────────────────────────────────────────────────────────
n_rows, n_cols = matrix.shape
for x in np.arange(-0.5, n_cols, 1):
    ax.axvline(x, color="white", linewidth=2.2, zorder=3)
for y in np.arange(-0.5, n_rows, 1):
    ax.axhline(y, color="white", linewidth=2.2, zorder=3)

# ── Cell text ─────────────────────────────────────────────────────────────────
for i in range(n_rows):
    for j in range(n_cols):
        val = matrix[i, j]
        norm_val = val / vmax
        color = "#F5F5F5" if norm_val > 0.50 else "#1C1C1C"
        weight = "bold" if bold_mask[i, j] else "normal"
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight=weight,
            color=color,
            zorder=4,
        )

# ── Tick labels ───────────────────────────────────────────────────────────────
ax.set_xticks(range(n_cols))
ax.set_xticklabels(methods, fontsize=13, fontweight="bold")
ax.set_yticks(range(n_rows))
ax.set_yticklabels(models, fontsize=13)
ax.tick_params(length=0, pad=7)
ax.xaxis.tick_top()

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.018)
cbar.set_label("Ratio (%)", fontsize=10, labelpad=9)
cbar.ax.tick_params(labelsize=9, length=3)
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig("degeneration_ratio.png", dpi=200, bbox_inches="tight", facecolor="white")
