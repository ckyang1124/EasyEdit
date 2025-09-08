import json
from argparse import ArgumentParser

def calculate_accuracy(input_file: str):
    track = input_file.split("/")[-1].split(".json")[0]
    loc_audio_num = 3 if track == "Gender" else 4
    data = json.load(open(input_file))
    
    acc = {
        "rel": {"acc": 0, "total": 0, "skipped": 0},
        "gen": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(3)],
        "port": {"acc": 0, "total": 0, "skipped": 0},
        "loc": {
            "audio": [{"acc": 0, "total": 0, "skipped": 0} for _ in range(loc_audio_num)],
            "text": {"acc": 0, "total": 0, "skipped": 0}
        }
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
        for j, loc_audio_eval in enumerate(item["locality_audio_evaluation"]["evaluations"]):
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

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the input JSON file containing evaluation results.")
    args = parser.parse_args()
    
    acc = calculate_accuracy(args.input_file)
    print(f"Result of {args.input_file}:")
    
    # Reliability
    rel_acc = 0.0 if acc["rel"]["total"] == 0 else acc["rel"]["acc"] / acc["rel"]["total"]
    print(f"- Reliability: {rel_acc:.6f} ({acc['rel']['acc']} / {acc['rel']['total']})")
    
    # Generality
    gen_all_acc = 0.0 if sum(g["total"] for g in acc["gen"]) == 0 else sum(g["acc"] for g in acc["gen"]) / sum(g["total"] for g in acc["gen"])
    print(f"- Generality (Overall): {gen_all_acc:.6f} ({sum(g['acc'] for g in acc['gen'])} / {sum(g['total'] for g in acc['gen'])})")
    for j, gen in enumerate(acc["gen"]):
        gen_acc = 0.0 if gen["total"] == 0 else gen["acc"] / gen["total"]
        print(f"  - Type {j+1}: {gen_acc:.6f} ({gen['acc']} / {gen['total']})")
    
    # Audio Locality
    audio_loc_all_acc = 0.0 if sum(a["total"] for a in acc["loc"]["audio"]) == 0 else sum(a["acc"] for a in acc["loc"]["audio"]) / sum(a["total"] for a in acc["loc"]["audio"])
    print(f"- Audio Locality (Overall): {audio_loc_all_acc:.6f} ({sum(a['acc'] for a in acc['loc']['audio'])} / {sum(a['total'] for a in acc['loc']['audio'])})")
    for j, loc_audio in enumerate(acc["loc"]["audio"]):
        loc_audio_acc = 0.0 if loc_audio["total"] == 0 else loc_audio["acc"] / loc_audio["total"]
        type_index = j + 1 if len(acc["loc"]["audio"]) == 4 else j + 1 if j == 0 else j + 2
        print(f"  - Type {type_index}: {loc_audio_acc:.6f} ({loc_audio['acc']} / {loc_audio['total']})")
    
    # Text Locality
    loc_text_acc = 0.0 if acc["loc"]["text"]["total"] == 0 else acc["loc"]["text"]["acc"] / acc["loc"]["text"]["total"]
    print(f"- Text Locality: {loc_text_acc:.6f} ({acc['loc']['text']['acc']} / {acc['loc']['text']['total']})")
    
    # Portability
    port_acc = 0.0 if acc["port"]["total"] == 0 else acc["port"]["acc"] / acc["port"]["total"]
    print(f"- Portability: {port_acc:.6f} ({acc['port']['acc']} / {acc['port']['total']})")
    
if __name__ == "__main__":
    main()