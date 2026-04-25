import json
import os


def parse_reliability_item(response_item: dict, original_item: dict) -> dict:
    """
    Returns:
    {
        "audio_path": str,
        "question": str,
        "original_answer": str,
        "model_response": str,
    }
    """

    if (
        response_item["reliability"]["audio_path"].split("/")[-1]
        != original_item["file"].split("/")[-1]
    ):
        raise ValueError(
            f"Audio path mismatch: {response_item['reliability']['audio_path']} vs {original_item['path']}"
        )

    if (
        response_item["reliability"]["question"]
        != original_item["reliability_question"]
    ):
        raise ValueError(
            f"Question mismatch: {response_item['reliability']['question']} vs {original_item['reliability_question']}"
        )

    return {
        "audio_path": response_item["reliability"]["audio_path"],
        "question": response_item["reliability"]["question"],
        "original_answer": original_item["original_answer"],
        "model_response": response_item["pre_edit"]["reliability"],
    }


def main():
    MODELS = ["AudioFlamingo3", "DeSTA", "Qwen"]
    CATEGORIES = ["Animal", "Emotion", "Gender", "Language"]
    DATASET_PATH = "../../../our_knowledge_editing/metadata/test/"

    for model in MODELS:
        for category in CATEGORIES:
            file_path = f"../EFK/{model}/single/{category}.json"
            model_response_data = json.load(open(file_path, "r"))
            original_data_path = os.path.join(
                DATASET_PATH, f"{category}_transcriptions_no_label.json"
            )
            original_data = json.load(open(original_data_path, "r"))

            parsed_data = [
                parse_reliability_item(response_item, original_item)
                for response_item, original_item in zip(
                    model_response_data, original_data
                )
            ]
            output_path = f"{model}/{category}.json"
            os.makedirs(model, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(parsed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
