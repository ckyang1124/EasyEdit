import argparse
import csv
import os

import numpy as np


METHODS = [
    "LMT",  # "FT (LLM)",
    "KE",
    "MEND",
    "UnKE",
    "WISE",
    "I-IKE",
    "IE-IKE",
    "PCT",  # "FT (Audio)",
]

TARGET_MODELS = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]
MODEL_DISPLAY = {
    "DeSTA2.5": "DeSTA",
    "Qwen2-Audio": "Qwen",
    "Audio Flamingo 3": "AF",
}

GEN_TYPE_COLS = [5, 6, 7]
LOC_TYPE_COLS = [9, 10, 11, 12]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Generality/Locality breakdown LaTeX table with per-group "
            "minimum cells highlighted."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../csvs/single_editing_all.csv"
        ),
        help="Path to single_editing_all.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output .tex file path. If empty, print to stdout.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=TARGET_MODELS + ["all"],
        help="Use one model or all models as table columns.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Breakdown of generality and audio locality evaluations across methods and models (\\%). The minimum (most challenging) scores within generality and audio locality for each model-method pair are highlighted."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:gen_loc_breakdown",
        help="LaTeX label for the table.",
    )
    parser.add_argument(
        "--model_as_row",
        action="store_true",
        help=(
            "Render model name as a standalone row block instead of occupying a "
            "dedicated Model column."
        ),
    )
    return parser.parse_args()


def parse_float(value):
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_table(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = {
        model: {
            method: {"gen": [np.nan] * 3, "loc": [np.nan] * 4} for method in METHODS
        }
        for model in TARGET_MODELS
    }

    current_model = None

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    for row in rows[2:]:
        if not row or len(row) < 13:
            continue

        model_name = row[0].strip()
        if model_name:
            current_model = model_name

        if current_model not in TARGET_MODELS:
            continue

        method_name = row[1].strip()
        if method_name == "FT (LLM)":
            method_name = "LMT"
        elif method_name == "FT (Audio)":
            method_name = "PCT"
        attr = row[2].strip()

        if attr != "ALL" or method_name not in METHODS:
            continue

        for i, col_idx in enumerate(GEN_TYPE_COLS):
            if col_idx < len(row):
                data[current_model][method_name]["gen"][i] = parse_float(row[col_idx])

        for i, col_idx in enumerate(LOC_TYPE_COLS):
            if col_idx < len(row):
                data[current_model][method_name]["loc"][i] = parse_float(row[col_idx])

    return data


def compute_min_mask(values):
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return [False] * len(values)

    min_val = float(np.min(arr[valid]))
    max_val = float(np.max(arr[valid]))
    if np.isclose(min_val, max_val, atol=1e-9, rtol=1e-9):
        return [False] * len(values)

    mask = np.isfinite(arr) & np.isclose(arr, min_val, atol=1e-9, rtol=1e-9)
    return mask.tolist()


def build_min_masks(data, models_to_use):
    masks = {}
    for model in models_to_use:
        masks[model] = {}
        for method in METHODS:
            gen_vals = data[model][method]["gen"]
            loc_vals = data[model][method]["loc"]
            masks[model][method] = {
                "gen": compute_min_mask(gen_vals),
                "loc": compute_min_mask(loc_vals),
            }
    return masks


def format_method_header(method):
    # if method == "LMT":
    #     return r"\makecell{\textbf{LMT}}"
    # if method == "PCT":
    #     return r"\makecell{\textbf{PCT}}"
    return method


def format_value(value):
    if not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def format_cell(value, highlight):
    text = format_value(value)
    if text == "--":
        return text
    if highlight:
        return rf"\cellcolor{{red!15}} {text}"
    return text


def build_latex_table(data, masks, models_to_use, caption, label, model_as_row=False):
    if model_as_row:
        col_def = "l|ccc|cccc"
        n_table_cols = 8
    else:
        col_def = "ll|ccc|cccc"
        n_table_cols = 9

    lines = []
    lines.append(
        "% Requires: \\usepackage{booktabs,multirow,makecell} and \\usepackage[table]{xcolor}"
    )
    lines.append(r"\begin{wraptable}{r}{0.5\textwidth}")
    lines.append(r"\vspace{-4mm}")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_def}}}")
    lines.append(r"\toprule")
    if model_as_row:
        lines.append(
            r"\multirow{2}{*}{\textbf{Method}} & "
            r"\multicolumn{3}{c|}{\textbf{Gen.}} & "
            r"\multicolumn{4}{c}{\textbf{Loc.}} \\"
        )
        lines.append(
            r" & \textbf{T-} & \textbf{A-} & \textbf{TA-} & \textbf{Inter-} & \textbf{Intra-} & \textbf{Tgt-} & \textbf{Cap-} \\"
        )
    else:
        lines.append(
            r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Method}} & "
            r"\multicolumn{3}{c|}{\textbf{Gen.}} & "
            r"\multicolumn{4}{c}{\textbf{Loc.}} \\"
        )
        lines.append(
            r" & & \textbf{T-} & \textbf{A-} & \textbf{TA-} & \textbf{Inter-} & \textbf{Intra-} & \textbf{Tgt-} & \textbf{Cap-} \\"
        )
    lines.append(r"\midrule")
    for model_idx, model in enumerate(models_to_use):
        model_disp = MODEL_DISPLAY.get(model, model)
        if model_as_row:
            lines.append(
                rf"\multicolumn{{{n_table_cols}}}{{l}}{{\textbf{{{model_disp}}}}} \\"
            )

        for method_idx, method in enumerate(METHODS):
            row_cells = []

            if not model_as_row:
                if method_idx == 0:
                    row_cells.append(
                        rf"\multirow{{{len(METHODS)}}}{{*}}{{{model_disp}}}"
                    )
                else:
                    row_cells.append("")

            row_cells.append(format_method_header(method))

            for type_idx in range(3):
                value = data[model][method]["gen"][type_idx]
                highlight = masks[model][method]["gen"][type_idx]
                row_cells.append(format_cell(value, highlight))

            for type_idx in range(4):
                value = data[model][method]["loc"][type_idx]
                highlight = masks[model][method]["loc"][type_idx]
                row_cells.append(format_cell(value, highlight))

            lines.append(" & ".join(row_cells) + r" \\")

        if model_idx < len(models_to_use) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{-5mm}")
    lines.append(r"\end{wraptable}")

    return "\n".join(lines)


def main():
    args = parse_args()
    data = read_table(args.csv_path)
    models_to_use = TARGET_MODELS if args.model == "all" else [args.model]
    masks = build_min_masks(data, models_to_use)

    latex_text = build_latex_table(
        data=data,
        masks=masks,
        models_to_use=models_to_use,
        caption=args.caption,
        label=args.label,
        model_as_row=args.model_as_row,
    )

    if args.output:
        output_path = os.path.abspath(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_text)
            f.write("\n")
        print(f"LaTeX table saved to {output_path}")
    else:
        print(latex_text)


if __name__ == "__main__":
    main()
