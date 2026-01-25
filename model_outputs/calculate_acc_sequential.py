import json
from argparse import ArgumentParser

EDITS_PER_SEQ = 10
NUM_SEQ = 10
AUDIO_LOC_TYPE2_TOTAL_BY_GAP = [90, 80, 70, 63, 53, 43, 33, 27, 17, 8]


def calculate_accuracy(input_file: str):
    """
    Calculate accruracy for a sequence.
    """
    data = json.load(open(input_file))

    acc_by_gap = [
        {
            "rel": {"acc": 0, "total": 0, "skipped": 0},
            "gen": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(3)],
            "port": {"acc": 0, "total": 0, "skipped": 0},
            "loc": {
                "audio": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(4)],
                "text": {"acc": 0, "total": 0, "skipped": 0},
            },
        }
        for _ in range(EDITS_PER_SEQ)
    ]

    assert (
        len(data["results"]) == EDITS_PER_SEQ
    ), f"{input_file}: Expected {EDITS_PER_SEQ} results, but got {len(data['results'])}."

    for i, edit_item in enumerate(data["results"]):
        # print("-")
        for j, item in enumerate(edit_item["judge_results"]):
            if j >= 5:
                continue
            gap = i - j
            # print(i, j, gap)

            # Reliability
            rel_eval = item["reliability_evaluation"]
            if (
                rel_eval["evaluations"][0]["skipped"]
                and rel_eval["evaluations"][0]["reason"] != "Missing response"
            ):
                acc_by_gap[gap]["rel"]["skipped"] += 1
                print(
                    f"Warning: Reliability evaluation skipped for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n"
                )
            elif rel_eval["evaluations"][0]["model_response"].strip() == "":
                acc_by_gap[gap]["rel"]["total"] += 1
                # print(f"Warning: Reliability evaluation has empty new output for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n")
            else:
                acc_by_gap[gap]["rel"]["total"] += 1
                if rel_eval["evaluations"][0]["correct"]:
                    if "efk" in input_file.lower() and gap == 4:
                        print(
                            f"Debug: Reliability evaluation for input file {input_file}, the {j}-th post-edit of the {i}-th edit (gap = {i - j})"
                        )
                        print(rel_eval)
                        print("---")
                    acc_by_gap[gap]["rel"]["acc"] += 1

            # Generality
            for j, gen_eval in enumerate(item["generality_evaluation"]["evaluations"]):
                if gen_eval["skipped"] and gen_eval["reason"] != "Missing response":
                    print(
                        f"Warning: Generality evaluation skipped for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n"
                    )
                    acc_by_gap[gap]["gen"][j]["skipped"] += 1
                elif gen_eval["model_response"].strip() == "":
                    acc_by_gap[gap]["gen"][j]["total"] += 1
                    # print(f"Warning: Generality evaluation has empty new output for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n")
                else:
                    acc_by_gap[gap]["gen"][j]["total"] += 1
                    if gen_eval["correct"]:
                        acc_by_gap[gap]["gen"][j]["acc"] += 1

            # Portability
            port_eval = item["portability_evaluation"]
            if (
                port_eval["evaluations"][0]["skipped"]
                and port_eval["evaluations"][0]["reason"] != "Missing response"
            ):
                print(
                    f"Warning: Portability evaluation skipped for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n"
                )
                acc_by_gap[gap]["port"]["skipped"] += 1
            elif port_eval["evaluations"][0]["model_response"].strip() == "":
                acc_by_gap[gap]["port"]["total"] += 1
                # print(f"Warning: Portability evaluation has empty new output for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n")
            else:
                acc_by_gap[gap]["port"]["total"] += 1
                if port_eval["evaluations"][0]["correct"]:
                    acc_by_gap[gap]["port"]["acc"] += 1

            # Locality - Audio
            if len(item["locality_audio_evaluation"]["evaluations"]) == 3:
                audio_loc_types = [0, 2, 3]  # Type 1, 3, 4
            else:
                audio_loc_types = [0, 1, 2, 3]  # Type 1, 2, 3, 4

            for j, loc_audio_eval in enumerate(
                item["locality_audio_evaluation"]["evaluations"]
            ):
                audio_loc_ind = audio_loc_types[j]
                if loc_audio_eval["skipped"]:
                    print(
                        f"Warning: Audio Locality evaluation skipped for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n"
                    )
                    acc_by_gap[gap]["loc"]["audio"][audio_loc_ind]["skipped"] += 1
                elif loc_audio_eval["new_output"].strip() == "":
                    acc_by_gap[gap]["loc"]["audio"][audio_loc_ind]["total"] += 1
                    # print(f"Warning: Audio Locality evaluation has empty new output for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n")
                else:
                    acc_by_gap[gap]["loc"]["audio"][audio_loc_ind]["total"] += 1
                    if loc_audio_eval["consistent"]:
                        acc_by_gap[gap]["loc"]["audio"][audio_loc_ind]["acc"] += 1

            # Locality - Text
            loc_text_eval = item["locality_text_evaluation"]
            if loc_text_eval["evaluations"][0]["skipped"]:
                print(
                    f"Warning: Text Locality evaluation skipped for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n"
                )
                acc_by_gap[gap]["loc"]["text"]["skipped"] += 1
            elif loc_text_eval["evaluations"][0]["new_output"].strip() == "":
                acc_by_gap[gap]["loc"]["text"]["total"] += 1
                # print(f"Warning: Text Locality evaluation has empty new output for input file {input_file}, the {j}-th post-edit of the {i}-th edit\n")
            else:
                acc_by_gap[gap]["loc"]["text"]["total"] += 1
                if loc_text_eval["evaluations"][0]["consistent"]:
                    acc_by_gap[gap]["loc"]["text"]["acc"] += 1
    return acc_by_gap


def update_global_acc(global_acc, acc):
    keys = ["acc", "total", "skipped"]
    for gap in range(EDITS_PER_SEQ):
        # Reliability
        for key in keys:
            global_acc[gap]["rel"][key] += acc[gap]["rel"][key]

        # Generality
        for j in range(3):
            for key in keys:
                global_acc[gap]["gen"][j][key] += acc[gap]["gen"][j][key]

        # Portability
        for key in keys:
            global_acc[gap]["port"][key] += acc[gap]["port"][key]

        # Locality - Audio
        for j in range(4):
            for key in keys:
                global_acc[gap]["loc"]["audio"][j][key] += acc[gap]["loc"]["audio"][j][
                    key
                ]

        # Locality - Text
        for key in keys:
            global_acc[gap]["loc"]["text"][key] += acc[gap]["loc"]["text"][key]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        required=True,
        help="Path to the input folder containing 10 sequential editing evaluation result.",
    )
    parser.add_argument(
        "--add_fixed_prefix",
        "--afp",
        action="store_true",
        help="Whether to add '_fixed' suffix to the input file names.",
    )
    args = parser.parse_args()

    global_acc_by_gap = [
        {
            "rel": {"acc": 0, "total": 0, "skipped": 0},
            "gen": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(3)],
            "port": {"acc": 0, "total": 0, "skipped": 0},
            "loc": {
                "audio": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(4)],
                "text": {"acc": 0, "total": 0, "skipped": 0},
            },
        }
        for _ in range(EDITS_PER_SEQ)
    ]

    for seq_id in range(NUM_SEQ):
        input_file = f"{args.input_folder}/{seq_id}_fixed.json"

        acc_by_gap = calculate_accuracy(input_file)
        update_global_acc(global_acc_by_gap, acc_by_gap)
        # print(f"Result of {input_file}:")

    print(f"Result for {args.input_folder}:")
    for gap, acc in enumerate(global_acc_by_gap):
        if gap > 5:
            continue
        print(f"=== Gap {gap} ===")
        all_accs = []

        # Reliability
        rel_acc = (
            0.0 if acc["rel"]["total"] == 0 else acc["rel"]["acc"] / acc["rel"]["total"]
        )
        all_accs.append(rel_acc)
        print(
            f"- Reliability: {rel_acc:.6f} ({acc['rel']['acc']} / {acc['rel']['total']})"
        )

        # Generality
        gen_all_acc = (
            0.0
            if sum(g["total"] for g in acc["gen"]) == 0
            else sum(g["acc"] for g in acc["gen"]) / sum(g["total"] for g in acc["gen"])
        )
        all_accs.append(gen_all_acc)
        print(
            f"- Generality (Overall): {gen_all_acc:.6f} ({sum(g['acc'] for g in acc['gen'])} / {sum(g['total'] for g in acc['gen'])})"
        )
        for j, gen in enumerate(acc["gen"]):
            gen_acc = 0.0 if gen["total"] == 0 else gen["acc"] / gen["total"]
            all_accs.append(gen_acc)
            print(f"  - Type {j+1}: {gen_acc:.6f} ({gen['acc']} / {gen['total']})")

        # Audio Locality
        audio_loc_all_acc = (
            0.0
            if sum(a["total"] for a in acc["loc"]["audio"]) == 0
            else sum(a["acc"] for a in acc["loc"]["audio"])
            / sum(a["total"] for a in acc["loc"]["audio"])
        )
        all_accs.append(audio_loc_all_acc)
        print(
            f"- Audio Locality (Overall): {audio_loc_all_acc:.6f} ({sum(a['acc'] for a in acc['loc']['audio'])} / {sum(a['total'] for a in acc['loc']['audio'])})"
        )
        assert (
            len(acc["loc"]["audio"]) == 4
        ), "Audio locality evaluations should have 4 types."
        # assert acc["loc"]["audio"][1]["total"] == AUDIO_LOC_TYPE2_TOTAL_BY_GAP[gap], f"Type 2 audio locality total count ({ acc['loc']['audio'][1]['total']}) mismatch (expected: {AUDIO_LOC_TYPE2_TOTAL_BY_GAP[gap]}) at gap {gap}."
        for j, loc_audio in enumerate(acc["loc"]["audio"]):
            loc_audio_acc = (
                0.0
                if loc_audio["total"] == 0
                else loc_audio["acc"] / loc_audio["total"]
            )
            all_accs.append(loc_audio_acc)
            print(
                f"  - Type {j + 1}: {loc_audio_acc:.6f} ({loc_audio['acc']} / {loc_audio['total']})"
            )

        # Text Locality
        loc_text_acc = (
            0.0
            if acc["loc"]["text"]["total"] == 0
            else acc["loc"]["text"]["acc"] / acc["loc"]["text"]["total"]
        )
        all_accs.append(loc_text_acc)
        print(
            f"- Text Locality: {loc_text_acc:.6f} ({acc['loc']['text']['acc']} / {acc['loc']['text']['total']})"
        )

        # Portability
        port_acc = (
            0.0
            if acc["port"]["total"] == 0
            else acc["port"]["acc"] / acc["port"]["total"]
        )
        all_accs.append(port_acc)
        print(
            f"- Portability: {port_acc:.6f} ({acc['port']['acc']} / {acc['port']['total']})"
        )
        print("\t".join([f"{a*100:.2f}" for a in all_accs]))


if __name__ == "__main__":
    main()
