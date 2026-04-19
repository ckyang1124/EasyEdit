import json
from argparse import ArgumentParser

def safe_percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def evaluate_overfitting_responses(judge_result: str, type_index: int = 1):
    """Evaluate overfitting for Audio Locality Type based on the judge results.
    type_index: 
    0 - Audio Locality Type 1
    1 - Audio Locality Type 2
    2 - Audio Locality Type 3
    3 - Audio Locality Type 4
    """
    with open(judge_result, "r") as f:
        data = json.load(f)

    overfitting_count = 0
    audio_loc_fail = 0
    
    if len(data["results"]) != 300:
        raise ValueError(f"Expected 300 evaluation items, got {len(data['results'])}")

    for item in data["results"]:
        audio_loc_eval = item["locality_audio_evaluation"]
        
        if audio_loc_eval["total_items"] != 4 and "gender" not in judge_result.lower():
            raise ValueError(f"Expected 4 audio locality evaluation items, got {audio_loc_eval['total_items']}")
        
        if audio_loc_eval["evaluations"][type_index]["consistent"]:
            continue

        audio_loc_fail += 1
        
        post_edit = item["post_edit"]
        reliability = post_edit["reliability"]
        if post_edit[f"locality_audio_type_{type_index}"].strip() == reliability.strip():
            overfitting_count += 1
    
    return overfitting_count, audio_loc_fail


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--detail", action="store_true", help="Print detailed results for each attribute")
    parser.add_argument("--type", type=int, default=2, choices=[1, 2, 3, 4], help="Audio Locality Type to evaluate (1, 2, 3, or 4)")
    args = parser.parse_args()
    
    METHODS = ["FT/last_layer", "FT/connector", "EFK", "MEND", "IKE_wo_examples", "IKE"]
    MODELS = ["DeSTA", "Qwen", "AudioFlamingo3"]
    ATTR = ["Animal", "Emotion", "Gender", "Language"]

    # judge_result = "./EFK/DeSTA/single/eval_result_202601/Animal.json"
    for model in MODELS:
        if args.detail:
            print(f"Evaluating overfitting for model: {model}")
        for method in METHODS:
            audio_loc_overfitting_count = 0
            global_audio_loc_fail = 0

            for attr in ATTR:
                judge_result = f"./{method}/{model}/single/eval_result_202601/{attr}.json"
                if args.type == 2 and attr == "Gender":
                    # Skip as gender does not have Audio Locality Type 2 evaluation.
                    continue
                
                if attr == "Gender" and args.type >= 3:
                    type_index = args.type - 2
                else:
                    type_index = args.type - 1
                    
                overfitting_count, audio_loc_fail = evaluate_overfitting_responses(judge_result, type_index=type_index)

                audio_loc_overfitting_count += overfitting_count
                global_audio_loc_fail += audio_loc_fail
                
                if args.detail:
                    print(f"  Attribute: {attr} - Audio Locality Type {args.type} Overfitting: {overfitting_count}/{audio_loc_fail} ({safe_percentage(overfitting_count, audio_loc_fail):.2f}%)")
                
            result_percentage = safe_percentage(audio_loc_overfitting_count, global_audio_loc_fail)
            if args.detail:
                print()
                print(f"Total Audio Locality Type {args.type} Overfitting for {method} on {model}: {audio_loc_overfitting_count}/{global_audio_loc_fail} ({result_percentage:.2f}%)")
                print()
            else:
                print(f"{result_percentage:.2f}", end="\t")
                if method == "MEND":
                    print("", end="\t")
            
        if args.detail:
            print("-" * 50)
        else:
            print()