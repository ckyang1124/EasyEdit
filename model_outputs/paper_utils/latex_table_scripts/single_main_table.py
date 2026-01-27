import csv
import sys
import os


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def format_value(value, all_values):
    if value is None:
        return "-"

    # Filter out None values for ranking
    valid_values = [v for v in all_values if v is not None]

    if not valid_values:
        return f"{value:.2f}"

    sorted_values = sorted(valid_values, reverse=True)
    best = sorted_values[0]
    second_best = sorted_values[1] if len(sorted_values) > 1 else -1.0

    formatted = f"{value:.2f}"

    if value == best:
        return f"\\textbf{{{formatted}}}"
    elif value == second_best:
        return f"\\underline{{{formatted}}}"
    else:
        return formatted


def main():
    csv_path = "EasyEdit/model_outputs/paper_utils/csvs/single_editing_all.csv"

    # Check if file exists in the relative path, otherwise try absolute path based on workspace info
    if not os.path.exists(csv_path):
        csv_path = "/home/biao/data/research_work/lalmke/EasyEdit/model_outputs/paper_utils/csvs/single_editing_all.csv"

    data = {}
    current_model = None

    # Methods order as they appear in the table columns
    methods_order = [
        "FT (LLM)",
        "FT (Audio)",
        "KE",
        "MEND",
        "UnKE",
        "I-IKE",
        "IE-IKE",
        "WISE",
    ]

    # Metrics map to column indices (0-based)
    # Rel: 3, Gen Avg: 4, Audio Loc Avg: 8, Text Loc: 13, Port: 14
    metrics_indices = {
        "Reliability": 3,
        "Generality (Avg.)": 4,
        "Audio Locality (Avg.)": 8,
        "Text Locality": 13,
        "Portability": 14,
    }

    metrics_order = [
        "Reliability",
        "Generality (Avg.)",
        "Audio Locality (Avg.)",
        "Text Locality",
        "Portability",
    ]

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
            attr = row[2].strip()

            if attr == "ALL":
                if method_name not in data[current_model]:
                    data[current_model][method_name] = {}

                for metric, idx in metrics_indices.items():
                    if idx < len(row):
                        data[current_model][method_name][metric] = parse_float(row[idx])
                    else:
                        data[current_model][method_name][metric] = None

    # Latex Header
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(
        r"\caption{The four metrics (\%) of the editing methods on the three models. Avg. indicates the average performance across all types of the corresponding metric. The best and second-best results are highlighted in \textbf{bold} and \underline{underlined}, respectively. {\color{red} Put New Values}}"
    )
    print("")
    print(r"% \resizebox{\textwidth}{!}{% 縮放表格以適應頁面寬度")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(
        r"\textbf{Metric} & \textbf{FT (LLM)} & \textbf{FT (Audio)} & \textbf{KE} & \textbf{MEND} & \textbf{UnKE} & \textbf{I-IKE} & \textbf{IE-IKE} & \textbf{WISE} \\"
    )
    print(r"\midrule")

    # Models order
    # Determine the model keys from data or hardcode order if needed.
    # The example has DeSTA2.5-Audio, Qwen2-Audio, Audio Flamingo 3
    # In CSV: 'DeSTA2.5', 'Qwen2-Audio', 'Audio Flamingo 3'

    model_mapping = {
        "DeSTA2.5": r"\textsc{\bfseries DeSTA2.5-Audio}",
        "Qwen2-Audio": r"\textsc{\bfseries Qwen2-Audio}",
        "Audio Flamingo 3": r"\textsc{\bfseries Audio Flamingo 3}",
    }

    target_models = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]

    for i, model_key in enumerate(target_models):
        if model_key not in data:
            continue

        display_name = model_mapping.get(model_key, model_key)

        print("")
        print(rf"\multicolumn{{9}}{{@{{}}l}}{{{display_name}}} \\")

        current_model_data = data[model_key]

        for metric in metrics_order:
            row_values = []
            values_for_ranking = []

            # First pass: collect values to determine ranking
            for method in methods_order:
                val = current_model_data.get(method, {}).get(metric)
                values_for_ranking.append(val)

            # Second pass: format string
            latex_cells = [metric]

            for method in methods_order:
                val = current_model_data.get(method, {}).get(metric)
                formatted = format_value(val, values_for_ranking)
                latex_cells.append(formatted)

            print(" & ".join(latex_cells) + r" \\")

        if i < len(target_models) - 1:
            print(r"\midrule")
        else:
            print(r"\bottomrule")

    print("")
    print(r"\end{tabular}%")
    print(r"% }")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
