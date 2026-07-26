"""
target_audio_based_portability.py

No-edit ground-truth control for the portability dimension. For each sample,
instead of asking the portability question against the (pre-edit) audio under
an edit/caption, we use the sample's own "target preservation" locality probe
(`locality.audio[-2]`), which is a real recording of the target (post-edit,
y_e) concept, and ask the exact same portability question with no system
prompt and no edit instructions at all. This measures each model's accuracy
on the portability questions when the target concept is the actual ground
truth, isolating "connected-reasoning task is hard" from "the edit didn't
propagate".
"""

from argparse import ArgumentParser
import json
import os
from typing import Optional

import librosa
from tqdm import tqdm

AUDIO_ROOT_TEMPLATE = "{dataset_root}/audio_data/{track}"


def resolve_audio_path(dataset_root: str, file_name: str, track: str) -> str:
    return os.path.join(
        AUDIO_ROOT_TEMPLATE.format(dataset_root=dataset_root, track=track), file_name
    )


class ModelWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name

        if model_name == "DeSTA":
            from desta import DeSTA25AudioModel

            self.model = DeSTA25AudioModel.from_pretrained(
                "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
            )
            self.model.to("cuda")
        elif model_name == "Qwen":
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

            model_id = "Qwen/Qwen2-Audio-7B-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", torch_dtype="bfloat16"
            )
        elif model_name == "AudioFlamingo3":
            from transformers import (
                AudioFlamingo3ForConditionalGeneration,
                AutoProcessor,
            )

            model_id = "nvidia/audio-flamingo-3-hf"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cuda", torch_dtype="bfloat16"
            )

    def generate(
        self, audio_path: str, transcription: Optional[str], question: str
    ) -> str:
        if self.model_name == "DeSTA":
            conversation = [
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{question}",
                    "audios": [{"audio": audio_path, "text": transcription}],
                },
            ]
            response = self.model.generate(
                conversation,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=256,
            )
            return response.text[0]
        elif self.model_name == "Qwen":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_path},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audio_array = librosa.load(
                audio_path, sr=self.processor.feature_extractor.sampling_rate
            )[0]
            inputs = self.processor(
                text=text,
                audio=[audio_array],
                return_tensors="pt",
                padding=True,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
            ).to(self.model.device)

            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=256,
            )
            generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return response
        elif self.model_name == "AudioFlamingo3":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": audio_path},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self.model.device, dtype=self.model.dtype)

            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=256,
            )
            response = self.processor.decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            return response[0]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["DeSTA", "Qwen", "AudioFlamingo3"],
        help="The model(s) to evaluate.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="../dataset/",
        help="The root directory of the dataset.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    for model_name in args.model:
        model = ModelWrapper(model_name)
        for track in ["Animal", "Emotion", "Gender", "Language"]:
            input_file = f"{args.dataset_root}/metadata/test/{track}_transcriptions_no_label.json"
            output_file = f"./model_outputs/target_audio_based_portability/{model_name}/single/{track}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(input_file) as f:
                data = json.load(f)

            results = []
            for item in tqdm(data, desc=f"Processing {model_name} - {track}"):
                # locality.audio[-2] is the "target preservation" probe: a real
                # recording of this sample's own edited_answer (y_e) concept.
                target_sample = item["locality"]["audio"][-2]
                assert (
                    target_sample["answer"].lower() == item["edited_answer"].lower()
                ), (
                    f"locality.audio[-2] answer {target_sample['answer']!r} does not match "
                    f"edited_answer {item['edited_answer']!r} for file {item['file']!r}"
                )

                audio_path = resolve_audio_path(
                    args.dataset_root, target_sample["file"], target_sample["track"]
                )
                question = item["portability"]["audio"]["question"]
                response = model.generate(
                    audio_path, target_sample.get("transcription"), question
                )

                results.append(
                    {
                        "file": item["file"],
                        "original_answer": item["original_answer"],
                        "edited_answer": item["edited_answer"],
                        "target_audio_file": target_sample["file"],
                        "portability_question": question,
                        "portability_answer": item["portability"]["audio"]["answer"],
                        "model_response": response,
                    }
                )

                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
