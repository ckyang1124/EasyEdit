import sys
import csv
import os
import re


def get_value(cell):
    # Extract the number before ' ('
    match = re.match(r"([\d\.]+)\s*\(", cell)
    if match:
        return match.group(1)
    if cell.strip() == "":
        return "-"
    return cell.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python detailed_seq.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    # Path to csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../csvs/seqential_editing_all.csv")

    data = {}  # Method -> Gap -> {Metric: Value}

    # Metrics indices in the CSV (based on analysis)
    # 3: Rel
    # 4: Gen (Avg)
    # 8: Audio Loc (Avg)
    # 13: Text Loc
    # 14: Port

    current_model = None
    current_method = None

    methods_order = [
        "FT (LLM)",
        "FT (Audio)",
        "MEND",
        "KE",
        "UnKE",
        "IE-IKE",
        "I-IKE",
        "WISE",
    ]

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

            for i in range(2, len(rows)):
                row = rows[i]
                if len(row) < 3:
                    continue

                model_col = row[0].strip()
                method_col = row[1].strip()
                gap_col = row[2].strip()

                if model_col:
                    current_model = model_col

                if method_col:
                    current_method = method_col

                if current_model != model_name:
                    continue

                if not gap_col.isdigit():
                    continue

                gap = int(gap_col)

                if current_method not in data:
                    data[current_method] = {}

                try:
                    rel = get_value(row[3])
                    gen = get_value(row[4])
                    audio_loc = get_value(row[8])
                    text_loc = get_value(row[13])
                    port = get_value(row[14])

                    data[current_method][gap] = {
                        "reliablity": rel,
                        "generality": gen,
                        "audio_locality": audio_loc,
                        "text_locality": text_loc,
                        "portability": port,
                    }
                except IndexError:
                    pass
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Generate Latex
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(
        f"\\caption{{Original result of the four metrics of different editing methods on {model_name}-Audio under sequential editing. For generality and audio locality, we present the averaged results. (\\%)}}"
    )
    print(r"\label{tab:desta_detailed_seq}")
    print(r"\resizebox{0.85\textwidth}{!}{%")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"    \begin{tabularx}{\textwidth}{Yc|Y|Y|Y|Y|Y}")
    print(r"    \toprule")
    print(
        r"    \textbf{Method} & \textbf{Gap} & \textbf{Reliability} & \textbf{Generality} & \textbf{Audio Locality} & \textbf{Text Locality} & \textbf{Portability} \\ \midrule"
    )

    # Count how many methods we are printing to know when to put bottomrule
    methods_to_print = [m for m in methods_order if m in data]

    for i, method in enumerate(methods_to_print):
        method_label = method
        if "FT (LLM)" in method:
            method_label = r"\makecell{FT\\(LLM)}"
        elif "FT (Audio)" in method:
            method_label = r"\makecell{FT\\(Audio)}"

        gaps = sorted(data[method].keys())
        num_rows = len(gaps)

        print(
            f"    \\multirow{{{num_rows}}}{{*}}{{\\textbf{{{method_label}}}}} ", end=""
        )

        for idx, gap in enumerate(gaps):
            vals = data[method][gap]
            prefix = "& " if idx == 0 else "     & "
            content = f"{gap} & {vals['reliablity']} & {vals['generality']} & {vals['audio_locality']} & {vals['text_locality']} & {vals['portability']} \\\\"

            if idx < num_rows - 1:
                content += " %\\cmidrule{2-7} "
            else:
                if i < len(methods_to_print) - 1:
                    content += " \\midrule"
                else:
                    content += " \\bottomrule"

            print(f"{prefix}{content}")

    print(r"    \end{tabularx}%")
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
