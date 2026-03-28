import csv
import sys
import os
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
        return "-"

    # Filter out None values for ranking
    valid_values = [v for v in all_values if v is not None]

    if not valid_values:
        return f"{value:.2f}"

    sorted_values = sorted(valid_values, reverse=True)
    best = sorted_values[0]
    second_best = sorted_values[1] if len(sorted_values) > 1 else -1.0

    formatted = f"{value:.2f}"
    ci95 = compute_ci95(value, n, use_wilson)
    if ci95 is not None:
        lower, upper = ci95
        formatted = f"{formatted} [{lower:.2f}, {upper:.2f}]"

    if value == best:
        return f"**{formatted}**"
    elif value == second_best:
        return f"<u>{formatted}</u>"
    else:
        return formatted


def main(args):
    use_wilson = args.use_wilson  # Flag to use Wilson score interval instead of Wald
    csv_path = "../csvs/single_editing_all.csv"

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
        r"Reliability (↑)": 3,
        r"Generality Avg. (↑)": 4,
        r"Audio Locality (Avg.) (↑)": 8,
        r"Text Locality (↑)": 13,
        r"Portability (↑)": 14,
    }

    metrics_order = [
        r"Reliability (↑)",
        r"Generality Avg. (↑)",
        r"Audio Locality (Avg.) (↑)",
        r"Text Locality (↑)",
        r"Portability (↑)",
    ]

    # Sample sizes for 95% confidence interval calculation.
    metric_sample_sizes = {
        r"Reliability (↑)": 1200,
        r"Generality Avg. (↑)": 3600,
        r"Audio Locality (Avg.) (↑)": 4500,
        r"Text Locality (↑)": 1200,
        r"Portability (↑)": 1200,
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
            attr = row[2].strip()

            if attr == "ALL":
                if method_name not in data[current_model]:
                    data[current_model][method_name] = {}

                for metric, idx in metrics_indices.items():
                    if idx < len(row):
                        data[current_model][method_name][metric] = parse_float(row[idx])
                    else:
                        data[current_model][method_name][metric] = None

    # Markdown Header
    print(
        "The four metrics (%) of the editing methods on the three models. "
        "Avg. indicates the average performance across all types of the corresponding metric. "
        "Each value is reported as mean ± 95% confidence interval. "
        "The best and second-best results are highlighted in bold and underlined, respectively."
    )
    print("")
    print(
        "| Model | Metric | FT (LLM) | FT (Audio) | KE | MEND | UnKE | I-IKE | IE-IKE | WISE |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    # Models order
    # Determine the model keys from data or hardcode order if needed.
    # The example has DeSTA2.5-Audio, Qwen2-Audio, Audio Flamingo 3
    # In CSV: 'DeSTA2.5', 'Qwen2-Audio', 'Audio Flamingo 3'

    model_mapping = {
        "DeSTA2.5": "**DeSTA2.5-Audio**",
        "Qwen2-Audio": "**Qwen2-Audio**",
        "Audio Flamingo 3": "**Audio Flamingo 3**",
    }

    target_models = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]

    for i, model_key in enumerate(target_models):
        if model_key not in data:
            continue

        display_name = model_mapping.get(model_key, model_key)

        current_model_data = data[model_key]

        for metric_idx, metric in enumerate(metrics_order):
            row_values = []
            values_for_ranking = []

            # First pass: collect values to determine ranking
            for method in methods_order:
                val = current_model_data.get(method, {}).get(metric)
                values_for_ranking.append(val)

            # Second pass: format string
            row_model = display_name if metric_idx == 0 else ""
            markdown_cells = [row_model, metric]

            for method in methods_order:
                val = current_model_data.get(method, {}).get(metric)
                formatted = format_value(
                    val,
                    values_for_ranking,
                    metric_sample_sizes.get(metric),
                    use_wilson=use_wilson,
                )
                markdown_cells.append(formatted)

            print("| " + " | ".join(markdown_cells) + " |")

        if i < len(target_models) - 1:
            print("|  |  |  |  |  |  |  |  |  |  |")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Markdown table from CSV data")
    parser.add_argument(
        "--use-wilson",
        action="store_true",
        help="Use Wilson score interval instead of Wald",
    )
    args = parser.parse_args()
    main(args)
