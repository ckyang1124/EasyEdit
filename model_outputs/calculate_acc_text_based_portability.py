import json
from argparse import ArgumentParser
import os


def calculate_accuracy(input_file: str):
    data = json.load(open(input_file))

    acc = {"acc": 0, "total": 0, "skipped": 0}

    for item in data["results"]:
        if item["skipped"]:
            acc["skipped"] += 1
        else:
            acc["total"] += 1
            if item["correct"]:
                acc["acc"] += 1

    return acc


def main(input_file):
    acc = calculate_accuracy(input_file)
    print("\n" + "=" * 50 + f"\nResult of {input_file}:")

    port_acc = 0.0 if acc["total"] == 0 else acc["acc"] / acc["total"] * 100.0
    print(f"- Portability: {port_acc:.2f} ({acc['acc']} / {acc['total']}, skipped {acc['skipped']})")

    return port_acc, acc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSON file or directory containing evaluation results.",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        main(args.input_file)

    elif os.path.isdir(args.input_file):
        all_outputs = []
        all_accs = []
        all_filenames = sorted(
            [
                filename
                for filename in os.listdir(args.input_file)
                if filename.endswith(".json")
            ]
        )
        for filename in all_filenames:
            input_path = os.path.join(args.input_file, filename)
            port_acc, acc = main(input_path)
            all_outputs.append(port_acc)
            all_accs.append(acc)

        print("\n----------------\nSummary of all tracks:")
        for filename, port_acc in zip(all_filenames, all_outputs):
            track = filename.split(".json")[0]
            print(f"{track}:\t{port_acc:.2f}")

        total_acc = {"acc": 0, "total": 0, "skipped": 0}
        for acc in all_accs:
            total_acc["acc"] += acc["acc"]
            total_acc["total"] += acc["total"]
            total_acc["skipped"] += acc["skipped"]

        overall_acc = (
            0.0
            if total_acc["total"] == 0
            else total_acc["acc"] / total_acc["total"] * 100.0
        )
        print("\n" + "=" * 50 + "\nOverall Result:")
        print(
            f"- Portability: {overall_acc:.2f} ({total_acc['acc']} / {total_acc['total']}, skipped {total_acc['skipped']})"
        )

        print()
        print("Output values (for easy copy-paste to Sheet):")
        print(
            "\t".join(f"{v:.2f}" for v in all_outputs) + f"\t{overall_acc:.2f}"
        )

    else:
        raise ValueError(
            f"Input path {args.input_file} is neither a file nor a directory."
        )

    print("-" * 50)
