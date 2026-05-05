import transformers
import torch
import os
import struct
from typing import Dict, List, Optional, Tuple

from DeSTA25_Audio.desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions
from DeSTA25_Audio.desta.utils.audio import AudioSegment
try:
    import librosa  # type: ignore
except ImportError:
    librosa = None

CONTEXT_TEMPLATES_CACHE = None

def find_sublist_start_index(list1, list2):
    for i in range(len(list1) - len(list2)+1):
        if all(a == b for a, b in zip(list1[i:i+len(list2)], list2)):
            return i
    return None

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

def print_trainable_parameters(model, new_weight, mask_ratio):
    original_parameters = 0
    new_weight_param = 0
    for _, param in new_weight.named_parameters():
        new_weight_param += param.numel()
    for _, param in model.named_parameters():
        original_parameters += param.numel()
    print(f"Original Model params: {original_parameters} || New Weight params: {new_weight_param} || trainable%: {100 * new_weight_param * (1-mask_ratio) / original_parameters}")


def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value

def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "./ckpts/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")
    
def get_params(model):
    return model.state_dict()

def get_shape(p, model): 
    # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

def get_logits(x):
    return x.logits if hasattr(x, "logits") else x

def tokenize(batch, tokenizer, device, context_templates=None, hparams=None):
    # Initialize lists to store the processed data from each batch entry
    len_temp = len(context_templates)
    prompts = [item['prompt'] for item in batch]
    labels = [item['target_new'] for item in batch]
    loc_prompts = [item['loc_prompt'] for item in batch]

    mask_token = -100  # ignore_index of CrossEntropyLoss
    if hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
        full_prompt = [tokenizer.apply_chat_template([{"role":"user", "content":templ.format(p)}],
                                        add_generation_prompt=True,
                                        tokenize=False) + ' ' + l
                        for templ in context_templates for p, l in zip(prompts, labels)]
        prompt_ids = tokenizer([tokenizer.apply_chat_template([{"role":"user", "content":templ.format(p)}],
                                    add_generation_prompt=True,
                                    tokenize=False) for templ in context_templates for p in prompts], return_tensors="pt", padding=True, truncation=True)["input_ids"]
    else:
        full_prompt = [f"{templ.format(p + ' ' + l)}" for templ in context_templates for p, l in zip(prompts, labels)]
        prompt_ids = tokenizer([f"{templ.format(p)}" for templ in context_templates for p in prompts], return_tensors="pt", padding=True, truncation=True)["input_ids"]
    full_prompt += loc_prompts  # add for subject activation

    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    # Mask the tokens based on hparams.objective_optimization
    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    act_masks = []
    deact_masks = []
    # Iterate through each batch entry and compute act_mask, deact_mask
    for i, loc_prompt in enumerate(loc_prompts):
        if loc_prompt in prompts[i]:  # subject: Factual Editing
            subject_token = tokenizer.encode(' ' + loc_prompt, add_special_tokens=False)
            subject_token1 = tokenizer.encode(loc_prompt, add_special_tokens=False)
            subject_length = len(subject_token)
            act_mask = torch.zeros_like(tokens['input_ids'][int(i*len_temp):int((i+1)*len_temp)])
            deact_mask = torch.zeros_like(tokens['input_ids'][int(i*len_temp):int((i+1)*len_temp)])
            for j, token in enumerate(tokens['input_ids'][int(i*len_temp):int((i+1)*len_temp)]):
                start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token)
                if start_idx is None:
                    start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token1)
                    subject_length = len(subject_token1)
                act_mask[j][start_idx: start_idx + subject_length] = 1
                deact_mask[j][:start_idx] = 1
                deact_mask[j][start_idx + subject_length:] = 1
        else:  # General Editing
            act_mask = None
            deact_mask = None

        # Append the masks to the lists
        act_masks.append(act_mask)
        deact_masks.append(deact_mask)

    # Convert to tensors and move to the specified device
    act_masks = [mask.to(device) if mask is not None else None for mask in act_masks]
    deact_masks = [mask.to(device) if mask is not None else None for mask in deact_masks]

    tokens = {key: val.to(device) for key, val in tokens.items()}
    # tokens:[(bs*(len_temp+1))*sequence_length],actmasks:bs*[len_temp*sequence_length],deact_masks:bs*[len_temp*sequence_length]
    return tokens, act_masks, deact_masks

def _ensure_desta_resources(model, tokenizer):
    if not hasattr(model, "tokenizer") or not hasattr(model, "processor"):
        model._setup_generation()
    tok = tokenizer or model.tokenizer
    return tok, model.processor


def _build_audio_context(
    tokenizer: transformers.PreTrainedTokenizer,
    question: str,
    audio_path: str,
    transcription: str,
    audio_locator: str,
    placeholder_token: str,
    prompt_size: int,
) -> Tuple[str, List[int]]:
    content = f"{audio_locator}\n{question}" if audio_locator not in question else question
    messages = [
        {
            "role": "user",
            "content": content,
            "audios": [
                {
                    "audio": audio_path,
                    "text": transcription if transcription is not None else " ",
                }
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace(audio_locator, f"<start_audio>{audio_locator}<end_audio>")
    token_list = tokenizer.tokenize(prompt)
    audio_size_list = [prompt_size]
    transcription_ids = tokenizer.encode(
        transcription if transcription is not None else " ", add_special_tokens=False, return_tensors="pt"
    )[0]
    transcription_size_list = [len(transcription_ids)]
    context_tokens, start_positions = _prepare_audio_context_and_start_positions(
        token_list=token_list,
        audio_locator=audio_locator,
        audio_size_list=audio_size_list,
        transcription_size_list=transcription_size_list,
        placeholder_token=placeholder_token,
    )
    context = tokenizer.convert_tokens_to_string(context_tokens)
    return context, start_positions


def _append_eos_token(text: str, tokenizer) -> str:
    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token or not text:
        return text
    return text if text.endswith(eos_token) else f"{text}{eos_token}"


def _build_audio_batch(
    model,
    tokenizer,
    requests: List[Dict],
    device,
    answer_key: str = "edited_answer",
    include_labels: bool = True,
    append_answer: bool = True,
):
    if not requests:
        return None, [], []

    tokenizer, processor = _ensure_desta_resources(model, tokenizer)
    dtype = next(model.parameters()).dtype

    contexts: List[str] = []
    start_positions_all: List[List[int]] = []
    audio_features = []
    transcriptions = []
    answers = []

    for req in requests:
        audio_path = req.get("audio_path")
        if audio_path is None or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        transcription = req.get("transcription", " ")
        context, start_positions = _build_audio_context(
            tokenizer=tokenizer,
            question=req.get("question", ""),
            audio_path=audio_path,
            transcription=transcription,
            audio_locator=model.audio_locator,
            placeholder_token=model.placeholder_token,
            prompt_size=model.config.prompt_size,
        )
        contexts.append(context)
        start_positions_all.append(start_positions)
        audio_features.append(AudioSegment.from_file(audio_path, target_sr=16000, channel_selector="average").samples)
        transcriptions.append(transcription if transcription is not None else " ")
        if append_answer and answer_key is not None:
            answer = req.get(answer_key, "") or ""
            if answer_key == "edited_answer":
                answer = _append_eos_token(answer, tokenizer)
        else:
            answer = ""
        answers.append(answer)

    full_sequences = []
    for context, answer in zip(contexts, answers):
        if answer.strip():
            separator = "" if context.endswith(" ") else " "
            full_sequences.append(context + separator + answer)
        else:
            full_sequences.append(context)
    
    audio_text_inputs = tokenizer(
        full_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_length=True,
        add_special_tokens=False,
    )
    audio_context_inputs = tokenizer(
        contexts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_length=True,
        add_special_tokens=False,
    )

    labels = (
        torch.full_like(audio_text_inputs["input_ids"], -100, dtype=torch.long) if include_labels else None
    )
    answer_lengths: List[int] = []
    last_prompt_token_loc: List[int] = []
    answer_exists = []
    
    seq_len = audio_text_inputs["input_ids"].size(1)
    ctx_seq_len = audio_context_inputs["input_ids"].size(1)
    
    for idx in range(audio_text_inputs["input_ids"].size(0)):
        text_length = audio_text_inputs["length"][idx].item()
        ctx_length = audio_context_inputs["length"][idx].item()
        pad_len_text = seq_len - text_length
        pad_len_ctx = ctx_seq_len - ctx_length

        start_answer = pad_len_text + ctx_length
        ans_len = max(text_length - ctx_length, 0)
        answer_lengths.append(ans_len)
        last_prompt_token_loc.append(max(start_answer - 1, 0))
        answer_exists.append(ans_len > 0)

        if labels is not None and ans_len > 0:
            labels[idx, start_answer:start_answer + ans_len] = audio_text_inputs["input_ids"][
                idx, start_answer:start_answer + ans_len
            ]

        start_positions_all[idx] = [(idx, pos + pad_len_ctx) for pos in start_positions_all[idx]]

    features_tensor = processor(audio_features, sampling_rate=16000, return_tensors="pt").input_features.to(
        device=device, dtype=dtype
    )
    batch_transcription_ids = [
        tokenizer.encode(trans, add_special_tokens=False, return_tensors="pt").long().to(device)
        for trans in transcriptions
    ]

    inputs = {
        "input_ids": audio_text_inputs["input_ids"].to(device),
        "attention_mask": audio_text_inputs["attention_mask"].to(device),
        "batch_features": features_tensor,
        "batch_transcription_ids": batch_transcription_ids,
        "batch_start_positions": [pos for positions in start_positions_all for pos in positions],
    }
    if include_labels and labels is not None:
        inputs["labels"] = labels.to(device)

    return inputs, answer_lengths, last_prompt_token_loc


def _load_audio_for_processor(audio_path: str, target_sr: int):
    if librosa is None:
        raise ImportError("librosa is required for Qwen audio preprocessing. Install it with `pip install librosa`.")
    if audio_path is None or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    wav, sr = librosa.load(audio_path, sr=target_sr)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def _build_qwen_audio_batch(
    processor,
    requests: List[Dict],
    device,
    include_labels: bool = True,
):
    if not requests:
        return None
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor must provide a tokenizer when building Qwen audio batches.")
    sampling_rate = processor.feature_extractor.sampling_rate

    question_texts: List[str] = []
    full_texts: List[str] = []
    audio_arrays = []

    for req in requests:
        audio_path = req.get("audio_path")
        question = req.get("question", "")
        answer = (req.get("edited_answer", "") or "").strip()

        conversation_user = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
        question_text = processor.apply_chat_template(
            conversation_user, add_generation_prompt=True, tokenize=False
        )
        question_texts.append(question_text)

        if include_labels:
            conversation_full = conversation_user + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            ]
            full_text = processor.apply_chat_template(
                conversation_full, add_generation_prompt=False, tokenize=False
            )
        else:
            full_text = question_text
        full_texts.append(full_text)
        audio_arrays.append(_load_audio_for_processor(audio_path, sampling_rate))

    question_inputs = processor(
        text=question_texts,
        audio=audio_arrays,
        return_tensors="pt",
        padding=True,
        sampling_rate=sampling_rate,
    )
    full_inputs = processor(
        text=full_texts,
        audio=audio_arrays,
        return_tensors="pt",
        padding=True,
        sampling_rate=sampling_rate,
    )

    prefix_lengths = question_inputs["attention_mask"].sum(dim=1).tolist()

    full_inputs = {
        key: (val.to(device) if isinstance(val, torch.Tensor) else val)
        for key, val in full_inputs.items()
    }

    if include_labels:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels = full_inputs["input_ids"].clone()
        labels[labels == pad_id] = -100
        for idx, prefix_len in enumerate(prefix_lengths):
            labels[idx, : int(prefix_len)] = -100
        full_inputs["labels"] = labels

    return full_inputs


def _build_text_locality_inputs(tokenizer, prompts: List[str], device):
    if not prompts:
        return None
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    labels = torch.full_like(tokenized["input_ids"], -100, dtype=torch.long)
    tokenized["labels"] = labels
    return {key: val.to(device) for key, val in tokenized.items()}


def multimodal_tokenize(
    requests,
    tokenizer,
    model,
    device,
    context_templates=None,
    hparams=None,
    text_locality=None,
    audio_locality=None,
):
    """
    Prepare edit and locality batches for DeSTA-style audio + text editing.
    """
    if not requests:
        raise ValueError("At least one edit request is required.")

    # Build edit batch (audio question + edited answer) with supervision.
    edit_batch, _, _ = _build_audio_batch(
        model=model,
        tokenizer=tokenizer,
        requests=requests,
        device=device,
        answer_key="edited_answer",
        include_labels=True,
    )
    
    # Prepare text locality prompts for the base LLM (no audio features).
    text_locality_inputs = None
    if text_locality:
        text_loc_prompts = []
        for item in text_locality:
            question = (item.get("question") or "").strip()
            if question:
                text_loc_prompts.append(question)
        if text_loc_prompts:
            text_locality_inputs = _build_text_locality_inputs(tokenizer, text_loc_prompts, device)

    # Prepare optional audio locality batch (audio question + original answer) without labels.
    audio_locality_inputs = None
    if audio_locality:
        audio_locality_inputs, _, _ = _build_audio_batch(
            model=model,
            tokenizer=tokenizer,
            requests=audio_locality,
            device=device,
            answer_key=None,
            include_labels=True,
            append_answer=False,
        )

    return {
        "edit_batch": edit_batch,
        "text_locality_inputs": text_locality_inputs,
        "audio_locality_inputs": audio_locality_inputs,
    }


def multimodal_tokenize_qwen(
    requests,
    processor,
    model,
    device,
    text_locality=None,
    audio_locality=None,
):
    """
    Prepare edit and locality batches for Qwen2-Audio editing.
    """
    if not requests:
        raise ValueError("At least one edit request is required.")

    edit_batch = _build_qwen_audio_batch(
        processor=processor,
        requests=requests,
        device=device,
        include_labels=True,
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor must provide a tokenizer for text locality prompts.")

    text_locality_inputs = None
    if text_locality:
        text_loc_prompts = []
        for item in text_locality:
            question = (item.get("question") or "").strip()
            if question:
                text_loc_prompts.append(question)
        if text_loc_prompts:
            text_locality_inputs = _build_text_locality_inputs(tokenizer, text_loc_prompts, device)

    audio_locality_inputs = None
    if audio_locality:
        audio_locality_inputs = _build_qwen_audio_batch(
            processor=processor,
            requests=audio_locality,
            device=device,
            include_labels=False,
        )

    return {
        "edit_batch": edit_batch,
        "text_locality_inputs": text_locality_inputs,
        "audio_locality_inputs": audio_locality_inputs,
    }


class EarlyStopMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.pre = 0
        self.val = 1e9
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.pre = self.val
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def stop(self, ):
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02

class EditingMeanAct:
    """Computes and stores the average and current value"""

    def __init__(self, min_a=1e9):
        self.reset(min_a=min_a)

    def reset(self, min_a=1e9):
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.min_a = min_a

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.min_a = min(self.min_a, val)

    def mean_act(self):
        return self.avg
    def min_act(self):
        return self.min_a

def get_context_templates(model, tok, length_params, device):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = []
        prompt_tok = tok(
            ["I", "You", "Because", 'Yes', 'Q: '],
            padding=True,
            return_tensors="pt"
        ).to(device)
        for length, n_gen in length_params: 

            gen_token = model.llm_model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tok.eos_token_id,
            )
            CONTEXT_TEMPLATES_CACHE += tok.batch_decode(gen_token, skip_special_tokens=True)
        CONTEXT_TEMPLATES_CACHE = ['{}'] + [_ + ' {}' for _ in CONTEXT_TEMPLATES_CACHE]
        # print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
