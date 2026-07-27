import json
import os

tracks = ["Animal", "Emotion", "Gender", "Language"]
models = ["AudioFlamingo3"]  # ["DeSTA", "Qwen", "AudioFlamingo3"]
method = "IKE_prompt_variants"
template_alg = "MEND"  # supplies the field layout / metadata for everything except post_edit


def build_post_edit(raw_item: dict, template_post_edit: dict) -> dict:
    post_edit = {}
    for key in template_post_edit.keys():
        if key == "reliability":
            post_edit[key] = raw_item["reliability"]["model_response"]
        elif key.startswith("generality_type_"):
            idx = int(key.rsplit("_", 1)[1])
            post_edit[key] = raw_item["generality"][idx]["model_response"]
        elif key.startswith("locality_audio_type_"):
            idx = int(key.rsplit("_", 1)[1])
            post_edit[key] = raw_item["locality_audio"][idx]["model_response"]
        elif key == "locality_text":
            post_edit[key] = raw_item["locality_text"][0]["model_response"]
        elif key == "portability_audio":
            post_edit[key] = raw_item["portability"]["model_response"]
        else:
            raise KeyError(f"Unrecognized post_edit key: {key}")
    return post_edit


def convert_single(model: str, track: str, prompt_variant: int):
    template_path = f"./{template_alg}/{model}/single/{track}.json"
    raw_path = f"./{method}/{model}/prompt_variant_{prompt_variant}/single/{track}.json"
    out_path = f"./{method}/{model}/prompt_variant_{prompt_variant}/single/{track}_converted.json"

    template = json.load(open(template_path))
    raw = json.load(open(raw_path))

    assert len(template) == len(raw), (
        f"Length mismatch for {model}/{track}/prompt_variant_{prompt_variant}: "
        f"template={len(template)} raw={len(raw)}"
    )

    converted = []
    for i, (tmpl_item, raw_item) in enumerate(zip(template, raw)):
        # sanity check ordering is consistent between the two sources
        if tmpl_item["reliability"]["answer"] != raw_item["reliability"]["answer"]:
            raise AssertionError(
                f"Order mismatch at index {i} for {model}/{track}/prompt_variant_{prompt_variant}"
            )

        out_item = {}
        for key, value in tmpl_item.items():
            if key in ("pre_edit", "post_edit"):
                continue
            out_item[key] = value

        out_item["post_edit"] = build_post_edit(raw_item, tmpl_item["post_edit"])
        converted.append(out_item)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)
    print(f"wrote {out_path}")


def convert_all():
    for model in models:
        for track in tracks:
            for prompt_variant in range(4):
                convert_single(model, track, prompt_variant)


if __name__ == "__main__":
    convert_all()
