import json
from argparse import ArgumentParser
import re
from collections import defaultdict
import os

track = None


class PortabilityClassification:
    def __init__(self, classification_dir: str):
        self.classification = self._load_portability_classification(classification_dir)

    def _load_portability_classification(self, base_dir: str):
        classification = {}
        for t in ["animal", "emotion", "gender", "language"]:
            file_path = f"{base_dir}/classified_{t}_questions.json"
            classification[t] = json.load(open(file_path, "r"))
        return classification

    def get_question_type(self, full_question: str):

        def _extract_question(full_question: str) -> str:
            return full_question.split("\nOptions")[0].strip()

        def _norm(s: str) -> str:
            return " ".join(s.split())

        question = _extract_question(full_question)
        q_norm = _norm(question)
        for t, types in self.classification.items():
            if track and t != track:
                continue
            for type_name, questions in types.items():
                for template in questions:
                    t_norm = _norm(template)
                    if t_norm == q_norm:
                        return type_name
                    if "[PLACEHOLDER]" in t_norm:
                        pattern = (
                            "^"
                            + re.escape(t_norm).replace(
                                re.escape("[PLACEHOLDER]"), r".+?"
                            )
                            + "$"
                        )
                        if re.fullmatch(pattern, q_norm, flags=re.DOTALL):
                            return type_name
        raise ValueError(
            f"Question '{full_question}' not found in the portability classification."
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSON file or directory containing judge result.",
    )
    parser.add_argument(
        "--classification_dir",
        type=str,
        default="../../our_knowledge_editing/portability/audio/classified_question/",
        help="Directory containing portability question types classification files.",
    )
    parser.add_argument(
        "--print_details",
        action="store_true",
        help="Whether to print detailed results for each question type.",
    )
    return parser.parse_args()


def process(input_file: str, classification_dir: str, print_details: bool):
    tracks = ["animal", "emotion", "gender", "language"]
    for t in tracks:
        if t in input_file:
            track = t
            break

    with open(input_file, "r") as f:
        judge_results = json.load(f)

    classifier = PortabilityClassification(classification_dir)

    results = defaultdict(lambda: {"total": 0, "correct": 0})
    for item in judge_results["results"]:
        full_question = (
            item["portability_question"]
            if "portability_question" in item
            else item["portability_audio"]["question"]
        )
        question_type = classifier.get_question_type(full_question)
        results[question_type]["total"] += 1
        if item["portability_evaluation"]["accuracy"] == 1.0:
            results[question_type]["correct"] += 1
        elif item["portability_evaluation"]["accuracy"] != 0.0:
            raise ValueError(
                f"Unexpected accuracy value: {item['portability_evaluation']['accuracy']}"
            )

    if print_details:
        print(f"Analysis results for {input_file}:\n")
    all_accs = []
    all_total = 0
    all_correct = 0
    for question_type, res in sorted(results.items()):
        total = res["total"]
        correct = res["correct"]
        accuracy = correct / total if total > 0 else 0.0
        all_accs.append(accuracy * 100)
        if print_details:
            print(
                f"Type: {question_type}, Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}"
            )
        all_total += total
        all_correct += correct
    print("\t".join([f"{acc:.2f}" for acc in all_accs]), end="\t")
    if print_details:
        avg_accuracy = all_correct / all_total if all_total > 0 else 0.0
        print(f"\nAvg: {avg_accuracy:.2%}")
        print(f"Total: {all_total}, Correct: {all_correct}")
        print("\n-----------------------------------")


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    classification_dir = args.classification_dir

    if os.path.isdir(input_file):
        for track_name in ["Animal", "Emotion", "Gender", "Language"]:
            file_path = os.path.join(input_file, f"{track_name}.json")
            process(file_path, classification_dir, args.print_details)
    else:
        process(input_file, classification_dir, args.print_details)
