import json

tracks = ["Animal", "Emotion", "Gender", "Language"]
models = ["AudioFlamingo3"] # ["DeSTA", "Qwen", "AudioFlamingo3"]
has_pre_edit_alg = "MEND"

def merge_single_pre_edit(method: str):
    for model in models:
        for track in tracks:
            has_pre_edit = json.load(open(f"./{has_pre_edit_alg}/{model}/single/{track}.json"))
            
            no_pre_edit = json.load(open(f"./{method}/{model}/single/{track}.json"))
            
            assert len(has_pre_edit) == len(no_pre_edit), f"Has pre-edit and no pre-edit data length mismatch for {method}, {track}, {model}, so cannot merge: {len(has_pre_edit)} vs {len(no_pre_edit)}"
            
            for i in range(len(has_pre_edit)):
                # assert all(
                #     has_pre_edit[i][key] == {
                #         k: v
                #         for k, v in no_pre_edit[i][key].items()
                #         if k != "original_answer"
                #     }
                #     for key in has_pre_edit[i].keys() 
                #     if key not in {"pre_edit", "post_edit"}
                # ), f"Mismatch at index {i} for track {track} and model {model}"
                
                
                for key in has_pre_edit[i].keys():
                    if key == "pre_edit" or key == "post_edit":
                        continue
                    
                    assert has_pre_edit[i][key] == no_pre_edit[i][key], f"Mismatch at index {i} for method {method} track {track} and model {model} on key {key}"
                
                no_pre_edit[i]["pre_edit"] = has_pre_edit[i]["pre_edit"]
            with open(f"./{method}/{model}/single/{track}.json", "w") as f:
                json.dump(no_pre_edit, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # merge_single_pre_edit("EFK")
    # merge_single_pre_edit("FT/last_layer")
    # merge_single_pre_edit("FT/connector")
    merge_single_pre_edit("IKE")
    merge_single_pre_edit("IKE_wo_examples")