import csv
import math
from argparse import ArgumentParser


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def compute_ci95(value, n, use_wilson=False):
    if value is None or n is None or n <= 0:
        return None

    p = max(0.0, min(1.0, value / 100.0))
    z = 1.96

    if use_wilson:
        denom = 1 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        spread = (z / denom) * math.sqrt((p * (1 - p)) / n + (z**2) / (4 * n**2))
        return (center - spread) * 100.0, (center + spread) * 100.0
    else:
        se = math.sqrt((p * (1.0 - p)) / n)
        spread = z * se
        return (p - spread) * 100.0, (p + spread) * 100.0


def format_value(value, all_values, n, use_wilson=False):
    if value is None:
        return r"\makecell{-}"

    # Filter out None values for ranking
    valid_values = [v for v in all_values if v is not None]

    if not valid_values:
        return f"\\makecell{{{value:.2f}}}"

    sorted_values = sorted(valid_values, reverse=True)
    best = sorted_values[0]
    second_best = sorted_values[1] if len(sorted_values) > 1 else -1.0

    value_text = f"{value:.2f}"
    ci_text = None
    ci95 = compute_ci95(value, n, use_wilson)
    if ci95 is not None:
        lower, upper = ci95
        # ci_text = f"{{\\tiny [{lower:.2f}, {upper:.2f}]}}"
        ci_text = f"{{\\fontsize{{5}}{{5.5}}\\selectfont [{lower:.2f}, {upper:.2f}]}}"

    if value == best:
        value_text = f"\\textbf{{{value_text}}}"
    elif value == second_best:
        value_text = f"\\underline{{{value_text}}}"

    if ci_text is not None:
        return f"\\makecell{{{value_text} \\\\[-4pt] {ci_text}}}"

    return f"\\makecell{{{value_text}}}"


def main(args):
    use_wilson = args.use_wilson  # Flag to use Wilson score interval instead of Wald
    if args.pre_edit_correctness:
        csv_path = "../csvs/single_editing_all_pre_edit_correct.csv"
    else:
        csv_path = "../csvs/single_editing_all.csv"

    data = {}
    current_model = None

    # Methods order as they appear in the table columns
    methods_order = [
        "LMT",  # "FT (LLM)",
        "KE",
        "MEND",
        "UnKE",
        "WISE",
        "I-IKE",
        "IE-IKE",
        "PCT",  # "FT (Audio)",
    ]

    # Metrics map to column indices (0-based)
    # Rel: 3, Gen Avg: 4, Audio Loc Avg: 8, Text Loc: 13, Port: 14
    metrics_indices = {
        r"Reliability": 3,
        r"Generality": 4,
        r"Audio Locality": 8,
        r"Text Locality": 13,
        r"Portability": 14,
        r"Edit Score": 15,
    }

    metrics_order = [
        r"Reliability",
        r"Generality",
        r"Audio Locality",
        r"Text Locality",
        r"Portability",
        r"Edit Score",
    ]

    # Sample sizes for 95% confidence interval calculation.
    if args.pre_edit_correctness:
        # print(
        #     "Using pre-edit correctness filtered sample sizes for confidence interval calculation."
        # )
        metric_sample_sizes_by_model = {
            "DeSTA2.5": {
                r"Reliability": 904,
                r"Generality": 2712,
                r"Audio Locality": 3349,
                r"Text Locality": 904,
                r"Portability": 904,
                r"Edit Score": 904,
            },
            "Qwen2-Audio": {
                r"Reliability": 984,
                r"Generality": 2952,
                r"Audio Locality": 3663,
                r"Text Locality": 984,
                r"Portability": 984,
                r"Edit Score": 984,
            },
            "Audio Flamingo 3": {
                r"Reliability": 989,
                r"Generality": 2967,
                r"Audio Locality": 3721,
                r"Text Locality": 989,
                r"Portability": 989,
                r"Edit Score": 989,
            },
        }
    else:
        metric_sample_sizes = {
            r"Reliability": 1200,
            r"Generality": 3600,
            r"Audio Locality": 4500,
            r"Text Locality": 1200,
            r"Portability": 1200,
            r"Edit Score": 1200,
        }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

        # Skip header rows
        # Row 0 is main header, Row 1 is sub header
        data_rows = rows[2:]

        for row in data_rows:
            if not row or len(row) < 2:
                continue

            model_name = row[0].strip()
            if model_name:
                current_model = model_name
                if current_model not in data:
                    data[current_model] = {}

            method_name = row[1].strip()
            if method_name == "FT (LLM)":
                method_name = "LMT"
            elif method_name == "FT (Audio)":
                method_name = "PCT"
            attr = row[2].strip()

            if attr == "ALL":
                if method_name not in data[current_model]:
                    data[current_model][method_name] = {}

                for metric, idx in metrics_indices.items():
                    if idx < len(row):
                        data[current_model][method_name][metric] = parse_float(row[idx])
                    else:
                        data[current_model][method_name][metric] = None

    for model in data:
        for method in data[model]:
            method_data = data[model][method]
            if method_data.get(r"Edit Score") is None:
                raise ValueError(
                    f"Missing Edit Score for {model} - {method}. Please ensure the CSV has the correct columns."
                )
                vals = [
                    method_data.get(r"Reliability"),
                    method_data.get(r"Generality"),
                    method_data.get(r"Audio Locality"),
                    method_data.get(r"Text Locality"),
                    method_data.get(r"Portability"),
                ]
                if all(v is not None for v in vals):
                    if any(v == 0.0 for v in vals):
                        method_data[r"Edit Score"] = 0.0
                    else:
                        method_data[r"Edit Score"] = len(vals) / sum(
                            1.0 / v for v in vals
                        )
                else:
                    method_data[r"Edit Score"] = None

    # LaTeX Header
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(
        r"\caption{Performance (\%; $\uparrow$) of editing methods on three models"
        + (
            " only considering samples with pre-edit correctness."
            if args.pre_edit_correctness
            else "."
        )
        + r" "
        r"Each value is reported as mean across all test samples with a 95\% confidence interval shown as [lower, upper]. "
        r"Generality and audio locality scores are averaged across all evaluation types and test samples. "
        r"Best and second-best results on individual metrics are shown in \textbf{bold} and \underline{underlined}, respectively.}"
    )
    if args.pre_edit_correctness:
        print(r"\label{tab:single_pre_edit_correct}")
    else:
        print(r"\label{tab:single_main}")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\resizebox{0.85\textwidth}{!}{%")
    print(r"\begin{tabular}{llcccccccc}")
    print(r"\toprule")
    print(
        r"\multirow{2.2}{*}{\textbf{Model}} "
        r"& \multirow{2.2}{*}{\textbf{Metric}} "
        r"& \multicolumn{5}{c}{\textbf{Edit LLM Backbone}} "
        r"& \multicolumn{3}{c}{\textbf{Frozen LLM Backbone}} \\"
    )
    print(r"\cmidrule(lr){3-7}\cmidrule(lr){8-10}")
    # print(
    #     r"\textbf{Model} & \textbf{Metric} & \textbf{FT (LLM)} & \textbf{FT (Audio)} & \textbf{KE} & \textbf{MEND} & \textbf{UnKE} & \textbf{I-IKE} & \textbf{IE-IKE} & \textbf{WISE} \\"
    # )
    model_row = r"& "
    for method in methods_order:
        model_row += f"& \\textbf{{{method}}} "
    model_row += r"\\"
    print(model_row)
    print(r"\midrule")

    # Models order
    # Determine the model keys from data or hardcode order if needed.
    # The example has DeSTA2.5-Audio, Qwen2-Audio, Audio Flamingo 3
    # In CSV: 'DeSTA2.5', 'Qwen2-Audio', 'Audio Flamingo 3'

    model_mapping = {
        "DeSTA2.5": r"\multirow{9.2}{*}{\textbf{DeSTA}} ",
        "Qwen2-Audio": r"\multirow{9.2}{*}{\textbf{Qwen}} ",
        "Audio Flamingo 3": r"\multirow{9.2}{*}{\textbf{AF}} ",
    }

    target_models = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]

    for i, model_key in enumerate(target_models):
        if model_key not in data:
            continue

        display_name = model_mapping.get(model_key, model_key)

        current_model_data = data[model_key]

        for metric_idx, metric in enumerate(metrics_order):
            values_for_ranking = []

            # First pass: collect values to determine ranking
            for method in methods_order:
                val = current_model_data.get(method, {}).get(metric)
                values_for_ranking.append(val)

            if metric == r"Edit Score":
                print(r"\cmidrule(lr){2-10}")

            # Second pass: format string
            row_model = display_name if metric_idx == 0 else ""
            latex_cells = [row_model, metric]

            for method in methods_order:
                val = current_model_data.get(method, {}).get(metric)

                if args.pre_edit_correctness:
                    sample_size = metric_sample_sizes_by_model.get(model_key, {}).get(
                        metric
                    )
                else:
                    sample_size = metric_sample_sizes.get(metric)

                formatted = format_value(
                    val,
                    values_for_ranking,
                    sample_size,
                    use_wilson=use_wilson,
                )
                latex_cells.append(formatted)

            print(" & ".join(latex_cells) + r" \\")

        if i < len(target_models) - 1:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table*}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate LaTeX table from CSV data")
    parser.add_argument(
        "--use-wilson",
        action="store_true",
        help="Use Wilson score interval instead of Wald",
    )
    parser.add_argument(
        "--pre-edit-correctness",
        action="store_true",
        help="Use CSV with pre-edit correctness filtering",
    )
    args = parser.parse_args()
    main(args)
