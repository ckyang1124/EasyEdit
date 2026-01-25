import argparse
import os
import json


def get_sequential_orig_output(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    pre_edit_all, data = data[0]["pre_edit_all"], data[1:]
    assert len(pre_edit_all) == len(
        data
    ), f"pre_edit_all and remaining data length mismatch for {input_file}: {len(pre_edit_all)} vs {len(data)}"

    with open(input_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    for pre_edit_item, item in zip(pre_edit_all, data):
        del item["post_edit"]
        item["pre_edit"] = pre_edit_item

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pre-edit outputs from sequential model output JSON file."
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSON file containing model outputs.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="Path to the output JSON file to save pre-edit outputs.",
    )

    args = parser.parse_args()
    get_sequential_orig_output(args.input_file, args.output_file)
