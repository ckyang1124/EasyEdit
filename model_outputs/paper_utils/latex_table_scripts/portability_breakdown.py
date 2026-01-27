import sys
import csv
import os


def get_value(cell):
    if cell.strip() == "" or cell.strip() == "#DIV/0!":
        return "0.00"  # Default for empty or error in this context usually implies N/A or 0. Template shows 0.00 for WISE.
    try:
        f = float(cell.strip())
        return f"{f:.2f}"
    except ValueError:
        return cell.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python portability_breakdown.py <attribute>")
        sys.exit(1)

    attribute = sys.argv[1].lower()

    # Configuration for attributes
    # name: (start_index, end_index) 0-based, inclusive
    # Indices based on analysis of the CSV
    # Animal: 2-9
    # Emotion: 10-13
    # Gender: 14-18
    # Language: 19-27

    attr_map = {
        "animal": {"range": (2, 9), "label": "Animal Sound", "tab_label": "animal"},
        "emotion": {"range": (10, 13), "label": "Emotion", "tab_label": "emotion"},
        "gender": {"range": (14, 18), "label": "Gender", "tab_label": "gender"},
        "language": {"range": (19, 27), "label": "Language", "tab_label": "language"},
    }

    if attribute not in attr_map:
        print(
            f"Unknown attribute: {attribute}. Supported: animal, emotion, gender, language"
        )
        sys.exit(1)

    config = attr_map[attribute]
    col_start, col_end = config["range"]
    num_data_cols = col_end - col_start + 1

    # Path to csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../csvs/portability_breakdown.csv")

    models_to_process = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]
    model_labels = {
        "DeSTA2.5": "DeSTA2.5-Audio",
        "Qwen2-Audio": "Qwen2-Audio",
        "Audio Flamingo 3": "Audio Flamingo 3",
    }

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

    data = {}  # Model -> Method -> [values]
    headers = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Row 2 contains the headers for attributes
            header_row = rows[1]
            headers = [h.strip() for h in header_row[col_start : col_end + 1]]

            # Data starts from row 2 (index 2)
            current_model = None

            for i in range(2, len(rows)):
                row = rows[i]
                if not row or len(row) < 2:
                    continue

                model_name = row[0].strip()
                method_name = row[1].strip()

                if model_name:
                    current_model = model_name

                if current_model not in data:
                    data[current_model] = {}

                # Extract values
                row_values = []
                for j in range(col_start, col_end + 1):
                    if j < len(row):
                        row_values.append(get_value(row[j]))
                    else:
                        row_values.append("0.00")

                data[current_model][method_name] = row_values

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Generate LaTeX
    print(r"\begin{table}")
    print(r"\centering")
    print(
        f"\\caption{{Breakdown of portability performance across different concept categories for the \\textit{{{config['label']}}} attribute. (\\%)}}"
    )
    print(f"\\label{{tab:port_{config['tab_label']}}}")
    print("")
    print(r"\resizebox{\textwidth}{!}{%")

    # Build tabular format
    # cc|cccc...
    col_def = "cc|" + "c" * num_data_cols
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print(r"\toprule")

    # Build header row
    header_str = r"\textbf{Model} & \textbf{Method}"
    for h in headers:
        header_str += f" & \\textbf{{{h}}}"
    header_str += r" \\ \midrule"
    print(header_str)
    print("")

    for m_idx, model in enumerate(models_to_process):
        if model not in data:
            continue

        model_disp = model_labels.get(model, model)

        # Start multirow
        print(
            f"\\multirow[c]{{11.5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{model_disp}}}}}}}"
        )
        print("")

        for idx, method in enumerate(methods_order):
            if method not in data[model]:
                vals = ["0.00"] * num_data_cols
            else:
                vals = data[model][method]

            # Format method label
            method_label = f"\\textbf{{{method}}}"
            if "FT (LLM)" == method:
                method_label = r"\textbf{\makecell{FT \\ (LLM)}}"
            elif "FT (Audio)" == method:
                method_label = r"\textbf{\makecell{FT \\ (Audio)}}"

            # The template had FT (Audio) for the first one (FT LLM).
            # I will output the CORRECT label FT (LLM) as extracted from CSV logic
            # but formatted nicely like the template.

            row_str = f" & {method_label}"
            for v in vals:
                row_str += f" & {v}"
            row_str += " \\\\"

            print(row_str, end="")

            if idx < len(methods_order) - 1:
                print(r" \cmidrule{2-" + str(2 + num_data_cols) + "} ")
            else:
                print("")  # End of row

        if m_idx < len(models_to_process) - 1:
            print(r"\midrule")
            print("")
            print("")
        else:
            pass

    print(r" \bottomrule")
    print(r"\end{tabular}%")
    print("}")
    print("")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
