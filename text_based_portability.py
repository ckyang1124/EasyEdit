from argparse import ArgumentParser
import json
import os
from tqdm import tqdm

IKE_SYSTEM_PROMPT = """You are performing in-context knowledge editing, but only for questions about audio inputs.

From now on, when analyzing any audio clip, you must apply the following edits consistently:

EDITS (New facts):
{edits}

where each entry has the form `pre_edit` → `post_edit`.

Rules:
1. If your reasoning or prediction about an audio clip would normally lead to `pre_edit`, you must instead treat it as `post_edit`.
2. All properties, attributes, and facts that belong to `post_edit` must be applied consistently, as if the audio were actually from `post_edit`.
3. If the user’s question is unrelated to these edits, you should answer normally without making changes.
4. Always ensure your final answers are fully consistent with the edited mapping.
"""

caption_templates = {
    "Gender": "The gender of the speaker in the speech clip is [answer].",
    "Language": "The language spoken in the speech clip is [answer].",
    "Emotion": "The speaker is [answer].",
    "Animal": "The animal making the sound is [answer].",
}

query_template = f"""Audio/Speech information: 
[caption]
Based on the information, answer the following question:
[Instruction]"""


class ModelWrapper:
    def __init__(self, model_name: str):
        # init model and tokenizer here
        self.model_name = model_name

        if model_name == "DeSTA":
            from desta import DeSTA25AudioModel

            # Load the model from Hugging Face
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

    def generate(self, item: dict, track: str, use_post_edit: bool = False) -> str:
        pre_edit = item["original_answer"]
        post_edit = item["edited_answer"]

        if use_post_edit:
            system_prompt = ""
            
            caption = caption_templates[track].replace("[answer]", post_edit)
            question = item["portability"]["audio"]["question"]
            prompt = query_template.replace("[caption]", caption).replace(
                "[Instruction]", question
            )
        else:
            edits = f"{pre_edit} → {post_edit}"
            system_prompt = IKE_SYSTEM_PROMPT.format(edits=edits)

            caption = caption_templates[track].replace("[answer]", pre_edit)
            question = item["portability"]["audio"]["question"]
            prompt = query_template.replace("[caption]", caption).replace(
                "[Instruction]", question
            )

        if self.model_name == "DeSTA":
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
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
            # Text-only conversation (no audio content) using the audio-analysis chat format
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=256,
            )
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return response
        elif self.model_name == "AudioFlamingo3":
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
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
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
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
    parser.add_argument(
        "--use-post-edit",
        action="store_true",
        help="Whether to use the post-edit target in the prompt without explaining the edits. If not set, the model will be prompted with the original answer and the edits.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    for model_name in args.model:
        model = ModelWrapper(model_name)
        for track in ["Animal", "Emotion", "Gender", "Language"]:
            input_file = (
                f"{args.dataset_root}/metadata/test/{track}_transcriptions_no_label.json"
            )
            if args.use_post_edit:
                output_file = f"./model_outputs/text_based_portability_use_post_edit/{model_name}/single/{track}.json"
            else:
                output_file = f"./model_outputs/text_based_portability/{model_name}/single/{track}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(input_file) as f:
                data = json.load(f)

            results = []
            for item in tqdm(data, desc=f"Processing {model_name} - {track}"):
                response = model.generate(item, track, args.use_post_edit)
                results.append(
                    {
                        "file": item["file"],
                        "original_answer": item["original_answer"],
                        "edited_answer": item["edited_answer"],
                        "portability_question": item["portability"]["audio"][
                            "question"
                        ],
                        "portability_answer": item["portability"]["audio"]["answer"],
                        "model_response": response,
                    }
                )

                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
