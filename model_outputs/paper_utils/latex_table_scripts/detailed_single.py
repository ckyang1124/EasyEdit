import sys
import pandas as pd
import os


def format_value(val):
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    if s == "N/A":
        return "-"
    # Try to float conversion for consistent formatting
    try:
        f = float(s)
        return "{:.2f}".format(f)
    except ValueError:
        return s


def main():
    if len(sys.argv) != 2:
        print("Usage: python detailed_single.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    # Path to CSV
    # Assuming script is in EasyEdit/model_outputs/paper_utils/latex_table_scripts/
    # And csv is in EasyEdit/model_outputs/paper_utils/csvs/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(
        os.path.join(base_dir, "../csvs/single_editing_all.csv")
    )

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Read CSV
    # The csv structure is a bit complex with headers.
    # Row 0: Headers 1
    # Row 1: Headers 2
    # Row 2+: Data
    # We load with header=None to handle manually
    df = pd.read_csv(csv_path, header=None)

    # Fill forward the Model column (Column 0)
    df[0] = df[0].ffill()
    # Fill forward the Method column (Column 1)
    df[1] = df[1].ffill()

    # Filter by model
    # Note: Column 0 contains model names like "DeSTA2.5", "Qwen2-Audio", etc.
    model_data = df[df[0] == model_name]

    if model_data.empty:
        # If no strict match, try fuzzy or stripping?
        # But user asks for args which is model name.
        print(f"No data found for model '{model_name}'")
        sys.exit(1)

    # Define the methods order and attributes order as per template
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

    attributes_order = ["ALL", "Animal", "Emotion", "Gender", "Language"]

    # Mapping of method name in CSV to Latex display name if special
    # But for lookup we use the CSV name.

    # Create a dictionary for quick lookup:  (method, attribute) -> [values...]
    # Values are columns 3 to 14 (inclusive indices, 0-based)
    # 3: Rel
    # 4-7: Gen (Avg, T1, T2, T3)
    # 8-12: Audio Loc (Avg, T1, T2, T3, T4)
    # 13: Text Loc
    # 14: Port

    lookup = {}
    for _, row in model_data.iterrows():
        method = str(row[1]).strip()
        attr = str(row[2]).strip()

        # Extract values
        vals = []
        for idx in range(3, 15):
            val = row[idx]

            # Special handling for Gender Audio Loc Type 2 (Column 10)
            if attr == "Gender" and idx == 10:
                vals.append("-")
            else:
                vals.append(format_value(val))

        lookup[(method, attr)] = vals

    # Header of Latex Table
    # Identify label name
    clean_name = model_name.lower().replace(" ", "").replace(".", "").replace("-", "")

    print(r"\begin{table}[ht!]")
    print()
    print(
        r"\caption{Detailed results of the four metrics of each auditory attribute across different editing methods on "
        + model_name
        + r"-Audio under single editing. Attr. denotes auditory attributes, and Port. denotes portability. For generality and audio locality, Avg. indicates the average performance across all types of the corresponding metric. (\%)}"
    )
    print(r"\label{tab:" + clean_name + r"-main}")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\centering")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"    \begin{tabular}{cc|c|cccc|ccccc|c|c}")
    print(r"    \toprule")
    print(
        r"    \multirow{2.4}{*}{\textbf{Method}} & \multirow{2.4}{*}{\textbf{Attr.}} & \multirow{2.4}{*}{\textbf{Reliability}} & \multicolumn{4}{c|}{\textbf{Generality}} & \multicolumn{5}{c|}{\textbf{Audio Locality}} & \multirow{2.4}{*}{\textbf{\makecell[c]{Text\\Locality}}} & \multirow{2.4}{*}{\textbf{Port.}} \\ [3pt]"
    )
    print(r"    % \cmidrule(lr){4-7} \cmidrule(lr){8-12}")
    print(
        r"    &  &  & Avg. & Type 1 & Type 2 & Type 3 & Avg. & Type 1 & Type 2 & Type 3 & Type 4 &  &  \\ "
    )
    print(r"    \midrule")

    for i, method in enumerate(methods_order):
        # Format method name for Latex
        if method == "FT (LLM)":
            method_latex = r"\multirow[c]{5.2}{*}{\textbf{\makecell{FT \\ (LLM)}}}"
        elif method == "FT (Audio)":
            method_latex = r"\multirow{5.2}{*}{\textbf{\makecell{FT \\ (Audio)}}}"
        else:
            method_latex = r"\multirow{5.2}{*}{\textbf{" + method + r"}}"

        print(r"    ")

        is_first_attr = True
        for attr in attributes_order:
            key = (method, attr)

            # Default empty values if missing
            vals = lookup.get(key, [""] * 12)

            # Construct row string
            row_str = "    "

            # First column: Method (only on first attribute)
            if is_first_attr:
                row_str += method_latex
                is_first_attr = False

            row_str += f" & {attr} & " + " & ".join(vals) + r" \\"

            if attr == "Language":
                row_str += r""
            else:
                row_str += r""

            print(row_str)

            # Add cmidrule after the ALL row
            if attr == "ALL":
                print(r"    \cmidrule(lr){2-14} ")
            # Optional comments can be added here if needed to strictly match user style

        if i < len(methods_order) - 1:
            print(r"    \midrule")

    print(r"    \bottomrule")
    print()
    print()
    print(r"    ")
    print(r"    \end{tabular}")
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
