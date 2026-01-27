import json
import statistics
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
                    # if "efk" in input_file.lower() and gap == 4:
                    #     print(
                    #         f"Debug: Reliability evaluation for input file {input_file}, the {j}-th post-edit of the {i}-th edit (gap = {i - j})"
                    #     )
                    #     print(rel_eval)
                    #     print("---")
                    # if "matches" in rel_eval["evaluations"][0]["reason"].lower():
                    #     print(
                    #         "Debug: Reliability evaluation 'matches' reason. It may be a wrong evaluation."
                    #     )
                    #     print(json.dumps(rel_eval, indent=2, ensure_ascii=False))
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
                        # if "matches" in gen_eval["reason"].lower():
                        #     print(
                        #         "Debug: Generality evaluation 'matches' reason. It may be a wrong evaluation."
                        #     )
                        #     print(json.dumps(gen_eval, indent=2, ensure_ascii=False))
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

    all_seq_accs = []
    for seq_id in range(NUM_SEQ):
        input_file = f"{args.input_folder}/{seq_id}_fixed.json"

        acc_by_gap = calculate_accuracy(input_file)
        update_global_acc(global_acc_by_gap, acc_by_gap)
        all_seq_accs.append(acc_by_gap)
        # print(f"Result of {input_file}:")

    final_output_strs = ""
    print(f"Result for {args.input_folder}:")
    for gap, acc in enumerate(global_acc_by_gap):
        if gap > 5:
            continue
        print(f"=== Gap {gap} ===")
        all_accs = []
        all_stds = []

        # Reliability
        rel_acc = (
            0.0 if acc["rel"]["total"] == 0 else acc["rel"]["acc"] / acc["rel"]["total"]
        )
        all_accs.append(rel_acc)

        rel_seq_accs = [
            (
                0.0
                if seq[gap]["rel"]["total"] == 0
                else seq[gap]["rel"]["acc"] / seq[gap]["rel"]["total"]
            )
            for seq in all_seq_accs
        ]
        rel_std = statistics.stdev(rel_seq_accs) if len(rel_seq_accs) > 1 else 0.0
        all_stds.append(rel_std)

        print(
            f"- Reliability: {rel_acc:.6f} +/- {rel_std:.6f} ({acc['rel']['acc']} / {acc['rel']['total']})"
        )
        print(f"\t All accs: {rel_seq_accs}")

        # Generality
        gen_all_acc = (
            0.0
            if sum(g["total"] for g in acc["gen"]) == 0
            else sum(g["acc"] for g in acc["gen"]) / sum(g["total"] for g in acc["gen"])
        )
        all_accs.append(gen_all_acc)

        gen_all_seq_accs = [
            (
                0.0
                if sum(g["total"] for g in seq[gap]["gen"]) == 0
                else sum(g["acc"] for g in seq[gap]["gen"])
                / sum(g["total"] for g in seq[gap]["gen"])
            )
            for seq in all_seq_accs
        ]
        gen_all_std = (
            statistics.stdev(gen_all_seq_accs) if len(gen_all_seq_accs) > 1 else 0.0
        )
        all_stds.append(gen_all_std)

        print(
            f"- Generality (Overall): {gen_all_acc:.6f} +/- {gen_all_std:.6f} ({sum(g['acc'] for g in acc['gen'])} / {sum(g['total'] for g in acc['gen'])})"
        )
        print(f"\t All accs: {gen_all_seq_accs}")
        for j, gen in enumerate(acc["gen"]):
            gen_acc = 0.0 if gen["total"] == 0 else gen["acc"] / gen["total"]
            all_accs.append(gen_acc)

            gen_seq_accs = [
                (
                    0.0
                    if seq[gap]["gen"][j]["total"] == 0
                    else seq[gap]["gen"][j]["acc"] / seq[gap]["gen"][j]["total"]
                )
                for seq in all_seq_accs
            ]
            gen_std = statistics.stdev(gen_seq_accs) if len(gen_seq_accs) > 1 else 0.0
            all_stds.append(gen_std)

            print(
                f"\t\t- Type {j+1}: {gen_acc:.6f} +/- {gen_std:.6f} ({gen['acc']} / {gen['total']})"
            )
            print(f"\t\t\t All accs: {gen_seq_accs}")

        # Audio Locality
        audio_loc_all_acc = (
            0.0
            if sum(a["total"] for a in acc["loc"]["audio"]) == 0
            else sum(a["acc"] for a in acc["loc"]["audio"])
            / sum(a["total"] for a in acc["loc"]["audio"])
        )
        all_accs.append(audio_loc_all_acc)

        audio_loc_seq_accs = [
            (
                0.0
                if sum(a["total"] for a in seq[gap]["loc"]["audio"]) == 0
                else sum(a["acc"] for a in seq[gap]["loc"]["audio"])
                / sum(a["total"] for a in seq[gap]["loc"]["audio"])
            )
            for seq in all_seq_accs
        ]
        audio_loc_std = (
            statistics.stdev(audio_loc_seq_accs) if len(audio_loc_seq_accs) > 1 else 0.0
        )
        all_stds.append(audio_loc_std)

        print(
            f"- Audio Locality (Overall): {audio_loc_all_acc:.6f} +/- {audio_loc_std:.6f} ({sum(a['acc'] for a in acc['loc']['audio'])} / {sum(a['total'] for a in acc['loc']['audio'])})"
        )
        print(f"\t All accs: {audio_loc_seq_accs}")
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

            loc_audio_seq_accs = [
                (
                    0.0
                    if seq[gap]["loc"]["audio"][j]["total"] == 0
                    else seq[gap]["loc"]["audio"][j]["acc"]
                    / seq[gap]["loc"]["audio"][j]["total"]
                )
                for seq in all_seq_accs
            ]
            loc_audio_std = (
                statistics.stdev(loc_audio_seq_accs)
                if len(loc_audio_seq_accs) > 1
                else 0.0
            )
            all_stds.append(loc_audio_std)

            print(
                f"  - Type {j + 1}: {loc_audio_acc:.6f} +/- {loc_audio_std:.6f} ({loc_audio['acc']} / {loc_audio['total']})"
            )
            print(f"\t\t All accs: {loc_audio_seq_accs}")

        # Text Locality
        loc_text_acc = (
            0.0
            if acc["loc"]["text"]["total"] == 0
            else acc["loc"]["text"]["acc"] / acc["loc"]["text"]["total"]
        )
        all_accs.append(loc_text_acc)

        loc_text_seq_accs = [
            (
                0.0
                if seq[gap]["loc"]["text"]["total"] == 0
                else seq[gap]["loc"]["text"]["acc"] / seq[gap]["loc"]["text"]["total"]
            )
            for seq in all_seq_accs
        ]
        loc_text_std = (
            statistics.stdev(loc_text_seq_accs) if len(loc_text_seq_accs) > 1 else 0.0
        )
        all_stds.append(loc_text_std)

        print(
            f"- Text Locality: {loc_text_acc:.6f} +/- {loc_text_std:.6f} ({acc['loc']['text']['acc']} / {acc['loc']['text']['total']})"
        )
        print(f"\t All accs: {loc_text_seq_accs}")

        # Portability
        port_acc = (
            0.0
            if acc["port"]["total"] == 0
            else acc["port"]["acc"] / acc["port"]["total"]
        )
        all_accs.append(port_acc)

        port_seq_accs = [
            (
                0.0
                if seq[gap]["port"]["total"] == 0
                else seq[gap]["port"]["acc"] / seq[gap]["port"]["total"]
            )
            for seq in all_seq_accs
        ]
        port_std = statistics.stdev(port_seq_accs) if len(port_seq_accs) > 1 else 0.0
        all_stds.append(port_std)

        print(
            f"- Portability: {port_acc:.6f} +/- {port_std:.6f} ({acc['port']['acc']} / {acc['port']['total']})"
        )
        print(f"\t All accs: {port_seq_accs}")
        output_strs = "\t".join(
            [f"{a*100:.2f} ({s*100:.2f})" for a, s in zip(all_accs, all_stds)]
        )
        print(output_strs)
        final_output_strs += f"{output_strs}\n"

    print("=== Final Summary ===")
    print(final_output_strs)


if __name__ == "__main__":
    main()
