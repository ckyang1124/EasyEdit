import json

tracks = ["Animal", "Emotion", "Gender", "Language"]
models = ["DeSTA"]
has_pre_edit_alg = "MEND"

def merge_single_pre_edit(method: str):
    for model in models:
        for track in tracks:
            has_pre_edit = json.load(open(f"./{has_pre_edit_alg}/{model}/single/{track}.json"))
            
            no_pre_edit = json.load(open(f"./{method}/{model}/single/{track}.json"))
            
            assert len(has_pre_edit) == len(no_pre_edit)
            for i in range(len(has_pre_edit)):
                assert all(
                    has_pre_edit[i][key] == no_pre_edit[i][key] 
                    for key in has_pre_edit[i].keys() 
                    if key not in {"pre_edit", "post_edit"}
                )
                no_pre_edit[i]["pre_edit"] = has_pre_edit[i]["pre_edit"]
            with open(f"./{method}/{model}/single/{track}.json", "w") as f:
                json.dump(no_pre_edit, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    merge_single_pre_edit("EFK")