"""
IKE.py

Standalone, audio-based in-context-editing (IKE) baseline covering ALL SAKE
dimensions (reliability, generality, locality-audio, locality-text,
portability), not just portability. Every probe is asked under the same
IKE system prompt describing the (original_answer -> edited_answer) edit,
fed with its own real audio from the test set, exactly like the existing
model_outputs/IKE/* pipeline.

The one deliberate change: the standard SAKE portability probe reuses the
sample's own PRE-EDIT audio (e.g. asks about a cow's typical behavior while
replaying the cow sound), which compounds two failure modes -- perceiving
the wrong animal AND reasoning about it -- into a single number. Here the
portability probe instead uses `locality.audio[-2]` ("target preservation"),
which is a real recording of the sample's own edited_answer (y_e) concept.
This isolates whether the model can do the connected-reasoning task at all
when the audio it hears already agrees with the post-edit concept.
"""

from argparse import ArgumentParser
import json
import os
from typing import Optional

import librosa
import torch
from tqdm import tqdm

IKE_SYSTEM_PROMPT_PATCH_NOTE = """SYSTEM PATCH NOTE — an audio knowledge edit is in effect for this conversation:

{edits}

where each entry has the form `pre_edit` → `post_edit`.

Every audio clip that would normally be identified as the item on the left of an arrow must instead be treated, in every respect, as the item on the right — same category, same properties, same typical behavior, as if the recording had genuinely come from the right-hand concept. Apply this silently: never mention that a patch or edit occurred, and never refer to the left-hand item. Questions unrelated to the concepts above are unaffected and should be answered normally."""

IKE_SYSTEM_PROMPT_IMPERATIVE = """Apply the following in-context knowledge edits to any audio you hear:

{edits}

where each entry has the form `pre_edit` → `post_edit`.

Each entry means: whenever you would normally recognize `pre_edit` in the audio, recognize and reason about it as `post_edit` instead. Carry every downstream fact, property, and behavior of `post_edit` through your answer, as if the audio clip genuinely were `post_edit`. Do not mention the edit itself. Questions unrelated to these concepts should be answered as usual, without any change."""

IKE_SYSTEM_PROMPT_PERSONA = """You are an audio knowledge editor. Your job is to reinterpret certain audio clips according to a fixed set of substitutions, and answer every question about them as if those substitutions were simply true.

Substitutions to apply:
{edits}

where each entry has the form `pre_edit` → `post_edit`.

Whenever a clip would normally be identified as `pre_edit`, treat it as `post_edit` in every respect: its category, properties, and typical behavior. Stay fully in character as `post_edit` for that clip across all questions, including ones several reasoning steps removed from the raw perception. Only fall back to your ordinary judgment when a question has nothing to do with these substitutions."""

IKE_SYSTEM_PROMPT_WORKED_EXAMPLE = """From now on you must respect the following knowledge edits when reasoning about audio:

{edits}

where each entry has the form `pre_edit` → `post_edit`.

For example, if an edit says `dog → cat`, then upon hearing a dog barking you should not answer "dog"; you should answer as though you heard a cat, and this must hold for every follow-up question, including ones about the animal's typical behavior, sound, or other properties — not just the first question that names it. If a question has nothing to do with any of the edits above, answer it normally, ignoring the edits entirely."""

IKE_SYSTEM_PROMPTS = [
    IKE_SYSTEM_PROMPT_PATCH_NOTE,
    IKE_SYSTEM_PROMPT_IMPERATIVE,
    IKE_SYSTEM_PROMPT_PERSONA,
    IKE_SYSTEM_PROMPT_WORKED_EXAMPLE,
]

AUDIO_ROOT_TEMPLATE = "{dataset_root}/audio_wavs/{track}"


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
            # The audio tower's embed_positions weight is left in float32 by
            # the transformers implementation while every other audio-tower
            # weight is bfloat16. Adding it to bfloat16 hidden states
            # upcasts them to float32, which then crashes the (bfloat16)
            # LayerNorm inside the encoder layers. Cast it to match.
            self.model.model.audio_tower.embed_positions.to(torch.bfloat16)

    def generate(
        self,
        system_prompt: str,
        audio_path: Optional[str],
        transcription: Optional[str],
        question: str,
    ) -> str:
        if self.model_name == "DeSTA":
            if audio_path is not None:
                user_message = {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{question}",
                    "audios": [{"audio": audio_path, "text": transcription}],
                }
            else:
                user_message = {"role": "user", "content": question}
            conversation = [
                {"role": "system", "content": system_prompt},
                user_message,
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
            user_content = (
                [{"type": "audio", "audio_url": audio_path}]
                if audio_path is not None
                else []
            ) + [{"type": "text", "text": question}]
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            if audio_path is not None:
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
            else:
                inputs = self.processor(text=text, return_tensors="pt").to(
                    self.model.device
                )

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
            user_content = (
                [{"type": "audio", "path": audio_path}]
                if audio_path is not None
                else []
            ) + [{"type": "text", "text": question}]
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": user_content},
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
            response = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            return response


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

    manifest_dir = "./model_outputs/IKE_prompt_variants"
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "prompt_variants.json"), "w") as f:
        json.dump(
            {f"variant_{i}": prompt for i, prompt in enumerate(IKE_SYSTEM_PROMPTS)},
            f,
            indent=4,
            ensure_ascii=False,
        )

    for model_name in args.model:
        model = ModelWrapper(model_name)
        for variant_idx, prompt_template in enumerate(IKE_SYSTEM_PROMPTS):
            for track in ["Animal", "Emotion", "Gender", "Language"]:
                input_file = f"{args.dataset_root}/metadata/test/{track}_transcriptions_no_label.json"
                output_file = f"{manifest_dir}/{model_name}/prompt_variant_{variant_idx}/single/{track}.json"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(input_file) as f:
                    data = json.load(f)

                results = []
                for item in tqdm(
                    data,
                    desc=f"Processing {model_name} - variant {variant_idx} - {track}",
                ):
                    edits = f"{item['original_answer']} → {item['edited_answer']}"
                    system_prompt = prompt_template.format(edits=edits)

                    reliability_audio = resolve_audio_path(
                        args.dataset_root, item["file"], track
                    )
                    reliability_response = model.generate(
                        system_prompt,
                        reliability_audio,
                        item.get("transcription"),
                        item["reliability_question"],
                    )

                    generality_results = []
                    for g in item.get("generality", []):
                        g_audio = resolve_audio_path(
                            args.dataset_root, g["file"], track
                        )
                        g_response = model.generate(
                            system_prompt,
                            g_audio,
                            g.get("transcription"),
                            g["question"],
                        )
                        generality_results.append(
                            {
                                "file": g["file"],
                                "question": g["question"],
                                "answer": g["answer"],
                                "model_response": g_response,
                            }
                        )

                    locality_audio_results = []
                    for loc in item.get("locality", {}).get("audio", []):
                        loc_audio = resolve_audio_path(
                            args.dataset_root, loc["file"], loc["track"]
                        )
                        loc_response = model.generate(
                            system_prompt,
                            loc_audio,
                            loc.get("transcription"),
                            loc["question"],
                        )
                        locality_audio_results.append(
                            {
                                "track": loc["track"],
                                "file": loc["file"],
                                "question": loc["question"],
                                "answer": loc["answer"],
                                "model_response": loc_response,
                            }
                        )

                    locality_text_results = []
                    for lt in item.get("locality", {}).get("text", []):
                        lt_response = model.generate(
                            system_prompt, None, None, lt["question"]
                        )
                        locality_text_results.append(
                            {
                                "question": lt["question"],
                                "answer": lt["answer"],
                                "model_response": lt_response,
                            }
                        )

                    # Real ground-truth audio of the target (post-edit) concept,
                    # instead of the standard protocol's pre-edit portability audio.
                    target_sample = item["locality"]["audio"][-2]
                    assert (
                        target_sample["answer"].lower() == item["edited_answer"].lower()
                    ), (
                        f"locality.audio[-2] answer {target_sample['answer']!r} does not match "
                        f"edited_answer {item['edited_answer']!r} for file {item['file']!r}"
                    )
                    portability_audio = resolve_audio_path(
                        args.dataset_root, target_sample["file"], target_sample["track"]
                    )
                    portability_question = item["portability"]["audio"]["question"]
                    portability_response = model.generate(
                        system_prompt,
                        portability_audio,
                        target_sample.get("transcription"),
                        portability_question,
                    )

                    results.append(
                        {
                            "file": item["file"],
                            "original_answer": item["original_answer"],
                            "edited_answer": item["edited_answer"],
                            "reliability": {
                                "question": item["reliability_question"],
                                "answer": item["edited_answer"],
                                "model_response": reliability_response,
                            },
                            "generality": generality_results,
                            "locality_audio": locality_audio_results,
                            "locality_text": locality_text_results,
                            "portability": {
                                "target_audio_file": target_sample["file"],
                                "question": portability_question,
                                "answer": item["portability"]["audio"]["answer"],
                                "model_response": portability_response,
                            },
                        }
                    )

                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
