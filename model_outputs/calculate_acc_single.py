import json
from argparse import ArgumentParser
import os


def calculate_accuracy(input_file: str):
    track = input_file.split("/")[-1].split(".json")[0]
    loc_audio_num = 3 if track == "Gender" else 4
    data = json.load(open(input_file))

    acc = {
        "rel": {"acc": 0, "total": 0, "skipped": 0},
        "gen": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(3)],
        "port": {"acc": 0, "total": 0, "skipped": 0},
        "loc": {
            "audio": [
                {"acc": 0, "total": 0, "skipped": 0} for _ in range(loc_audio_num)
            ],
            "text": {"acc": 0, "total": 0, "skipped": 0},
        },
    }

    for item in data["results"]:
        # Reliability
        rel_eval = item["reliability_evaluation"]
        if rel_eval["evaluations"][0]["skipped"]:
            acc["rel"]["skipped"] += 1
        else:
            acc["rel"]["total"] += 1
            if rel_eval["evaluations"][0]["correct"]:
                acc["rel"]["acc"] += 1

        # Generality
        for j, gen_eval in enumerate(item["generality_evaluation"]["evaluations"]):
            if gen_eval["skipped"]:
                acc["gen"][j]["skipped"] += 1
            else:
                acc["gen"][j]["total"] += 1
                if gen_eval["correct"]:
                    acc["gen"][j]["acc"] += 1

        # Portability
        port_eval = item["portability_evaluation"]
        if port_eval["evaluations"][0]["skipped"]:
            acc["port"]["skipped"] += 1
        else:
            acc["port"]["total"] += 1
            if port_eval["evaluations"][0]["correct"]:
                acc["port"]["acc"] += 1

        # Locality - Audio
        for j, loc_audio_eval in enumerate(
            item["locality_audio_evaluation"]["evaluations"]
        ):
            if loc_audio_eval["skipped"]:
                acc["loc"]["audio"][j]["skipped"] += 1
            else:
                acc["loc"]["audio"][j]["total"] += 1
                if loc_audio_eval["consistent"]:
                    acc["loc"]["audio"][j]["acc"] += 1

        # Locality - Text
        loc_text_eval = item["locality_text_evaluation"]
        if loc_text_eval["evaluations"][0]["skipped"]:
            acc["loc"]["text"]["skipped"] += 1
        else:
            acc["loc"]["text"]["total"] += 1
            if loc_text_eval["evaluations"][0]["consistent"]:
                acc["loc"]["text"]["acc"] += 1
    return acc


def main(input_file):

    acc = calculate_accuracy(input_file)
    print(f"Result of {input_file}:")

    outputs = []

    # Reliability
    rel_acc = (
        0.0
        if acc["rel"]["total"] == 0
        else acc["rel"]["acc"] / acc["rel"]["total"] * 100.0
    )
    print(f"- Reliability: {rel_acc:.2f} ({acc['rel']['acc']} / {acc['rel']['total']})")
    outputs.append(rel_acc)

    # Generality
    gen_all_acc = (
        0.0
        if sum(g["total"] for g in acc["gen"]) == 0
        else sum(g["acc"] for g in acc["gen"])
        / sum(g["total"] for g in acc["gen"])
        * 100.0
    )
    outputs.append(gen_all_acc)
    print(
        f"- Generality (Overall): {gen_all_acc:.2f} ({sum(g['acc'] for g in acc['gen'])} / {sum(g['total'] for g in acc['gen'])})"
    )
    for j, gen in enumerate(acc["gen"]):
        gen_acc = 0.0 if gen["total"] == 0 else gen["acc"] / gen["total"] * 100.0
        outputs.append(gen_acc)
        print(f"  - Type {j+1}: {gen_acc:.2f} ({gen['acc']} / {gen['total']})")

    # Audio Locality
    audio_loc_all_acc = (
        0.0
        if sum(a["total"] for a in acc["loc"]["audio"]) == 0
        else sum(a["acc"] for a in acc["loc"]["audio"])
        / sum(a["total"] for a in acc["loc"]["audio"])
        * 100.0
    )
    outputs.append(audio_loc_all_acc)
    print(
        f"- Audio Locality (Overall): {audio_loc_all_acc:.2f} ({sum(a['acc'] for a in acc['loc']['audio'])} / {sum(a['total'] for a in acc['loc']['audio'])})"
    )
    for j, loc_audio in enumerate(acc["loc"]["audio"]):
        loc_audio_acc = (
            0.0
            if loc_audio["total"] == 0
            else loc_audio["acc"] / loc_audio["total"] * 100.0
        )
        type_index = (
            j + 1 if len(acc["loc"]["audio"]) == 4 else j + 1 if j == 0 else j + 2
        )
        if len(acc["loc"]["audio"]) == 3 and j == 1:
            outputs.append("N/A")
        outputs.append(loc_audio_acc)
        print(
            f"  - Type {type_index}: {loc_audio_acc:.2f} ({loc_audio['acc']} / {loc_audio['total']})"
        )

    # Text Locality
    loc_text_acc = (
        0.0
        if acc["loc"]["text"]["total"] == 0
        else acc["loc"]["text"]["acc"] / acc["loc"]["text"]["total"] * 100.0
    )
    outputs.append(loc_text_acc)
    print(
        f"- Text Locality: {loc_text_acc:.2f} ({acc['loc']['text']['acc']} / {acc['loc']['text']['total']})"
    )

    # Portability
    port_acc = (
        0.0
        if acc["port"]["total"] == 0
        else acc["port"]["acc"] / acc["port"]["total"] * 100.0
    )
    outputs.append(port_acc)
    print(
        f"- Portability: {port_acc:.2f} ({acc['port']['acc']} / {acc['port']['total']})"
    )

    print()
    print("Output values (for easy copy-paste to Sheet):")
    print(
        "\t".join(
            f"{value:.2f}" if isinstance(value, float) else str(value)
            for value in outputs
        )
    )

    return outputs


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

    # if is a file:
    if os.path.isfile(args.input_file):
        main(args.input_file)
    # if is a directory:

    elif os.path.isdir(args.input_file):
        all_outputs = []
        all_filenames = sorted(
            [
                filename
                for filename in os.listdir(args.input_file)
                if filename.endswith(".json")
            ]
        )
        for filename in all_filenames:
            input_path = os.path.join(args.input_file, filename)
            all_outputs.append(main(input_path))

        print("\n----------------\nSummary of all files:")
        all_files_output_strs = []
        for filename, outputs in zip(all_filenames, all_outputs):
            output_str = "\t".join(
                f"{value:.2f}" if isinstance(value, float) else str(value)
                for value in outputs
            )
            print(f"{filename}:\n{output_str}")
            all_files_output_strs.append(output_str)

        # average each column
        print("\n----------------\nAverage across all files:")
        num_files = len(all_outputs)
        avg_outputs = []
        for col in range(len(all_outputs[0])):
            col_values = [
                all_outputs[row][col]
                for row in range(num_files)
                if isinstance(all_outputs[row][col], float)
            ]
            avg_value = sum(col_values) / len(col_values)
            avg_outputs.append(avg_value)
        avg_output_str = "\t".join(
            f"{value:.2f}" if isinstance(value, float) else str(value)
            for value in avg_outputs
        )
        print(avg_output_str)

        print(f"----------------")
        print(f"Easy paste. Average, {', '.join(all_filenames)}:")
        print(avg_output_str)
        for output_strs in all_files_output_strs:
            print(output_strs)

    else:
        raise ValueError(
            f"Input path {args.input_file} is neither a file nor a directory."
        )
