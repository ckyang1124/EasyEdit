import json
import os



def get_all_audio_locality_output(model: str):
    folder = f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/model_outputs/EFK/{model}/single/"
    
    all_locality = {}
    for attr in ["Animal", "Emotion", "Gender", "Language"]:
        file_path = os.path.join(folder, f"{attr}.json")
        data = json.load(open(file_path, "r"))
        
        for sample in data:
            for key in sample:
                if not key.startswith("locality_audio"):
                    continue
                
                audio_path = sample[key]["audio_path"].split("audio_wavs/")[-1]
                question = sample[key]["question"].strip()
                model_response = sample["pre_edit"][key].strip()
                
                all_locality[(audio_path, question)] = model_response
    
    return all_locality

if __name__ == "__main__":
    MODEL = "DeSTA"
    
    ALL_LOCALITY = get_all_audio_locality_output(model=MODEL)
    
    for i in range(10):
        metadata_path = f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{i}.json"
        model_orig_output_path = f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/model_outputs/seq_orig_model/{MODEL}/{i}.json"
        
        metadata = json.load(open(metadata_path, "r"))
        model_outputs = json.load(open(model_orig_output_path, "r"))
        for metadata_item, model_output_item in zip(metadata, model_outputs):
            for j in range(len(metadata_item["locality"]["audio"])):
                audio_loc_meta = metadata_item["locality"]["audio"][j]
                if "update" in audio_loc_meta and audio_loc_meta["update"]:
                    audio_path = audio_loc_meta["track"] + "/" + audio_loc_meta["file"]
                    question = audio_loc_meta["question"].strip()
                    transcription = audio_loc_meta["transcription"].strip()
                    answer = audio_loc_meta["answer"].strip()
                    
                    key = (audio_path, question)
                    correct_output = ALL_LOCALITY[key]
                    
                    model_output_item[f"locality_audio_type_{j}"] = {
                        "audio_path": "/work/b10902133/data/lalm-knowledge-editing/dataset/audio_wavs/" + audio_path,
                        "question": question,
                        "answer": answer,
                        "transcription": transcription,
                    }
                    model_output_item["pre_edit"][f"locality_audio_type_{j}"] = correct_output
                    
        save_path = f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/model_outputs/seq_orig_model/{MODEL}/{i}_fixed.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(model_outputs, open(save_path, "w"), indent=4, ensure_ascii=False)