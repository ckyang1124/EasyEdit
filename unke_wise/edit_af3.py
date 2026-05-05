#!/usr/bin/env python3
"""Knowledge editing for Audio-Flamingo-3 model.

Supports UNKE and WISE algorithms in single-edit and sequential modes.
Audio-Flamingo-3 uses a Whisper-style audio encoder with a Qwen2 language model
backbone (28 layers: 0-27).

Usage:
    python edit_af3.py --algorithm unke --category Language
    python edit_af3.py --algorithm wise --category Language
    python edit_af3.py --algorithm wise --mode sequential
"""
import argparse
import os
import random
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AudioFlamingo3ForConditionalGeneration

from editing_method.util import nethook

from edit_common import (
    set_seed,
    resolve_audio_path,
    load_preservation_data_raw,
    build_text_preserve,
    build_audio_preserve,
    load_metadata,
    load_groups_data,
    run_single_edits,
    run_sequential_edits,
    restore_model_weights,
    cleanup_cuda,
    ResultsWriter,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_TOKEN = ""
MODEL_NAME = "nvidia/audio-flamingo-3-hf"

GEN_KWARGS = dict(max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

def make_unke_hparams(category: str):
    from editing_method.algo.unke.unke_hparams import unkeHyperParams

    class _H(unkeHyperParams):
        def __init__(self):
            super().__init__(**{
                "model_name": MODEL_NAME,
                "alg_name": "unke",
                "ds_name": category,
                "layers": [20],
                "clamp_norm_factor": 4,
                "layer_selection": "all",
                "fact_token": "last",
                "ex_data_num": 20,
                "lr": 2e-4,
                "v_num_grad_steps": 25,
                "v_lr": 5e-1,
                "v_loss_layer": 27,
                "v_weight_decay": 1e-3,
                "optim_num_step": 50,
                "rewrite_module_tmp": "language_model.model.layers.{}.mlp.down_proj",
                "layer_module_tmp": "language_model.model.layers.{}",
                "mlp_module_tmp": "language_model.model.layers.{}.mlp",
                "attn_module_tmp": "language_model.model.layers.{}.self_attn",
                "ln_f_module": "language_model.model.norm",
                "lm_head_module": "language_model.lm_head",
                "arg_note": "audio-flamingo-3-qwen2",
            })

    return _H()


def make_wise_hparams(category: str, sequential: bool = False):
    from editing_method.algo.wise.wise_hparams import WISEHyperParams

    class _H(WISEHyperParams):
        def __init__(self):
            super().__init__(**{
                "model_name": MODEL_NAME,
                "ds_name": category,
                "mask_ratio": 0.2,
                "edit_lr": 1e-2,
                "n_iter": 100,
                "norm_constraint": 1.0,
                "alpha": 5.0,
                "beta": 20.0,
                "gamma": 10.0,
                "act_ratio": 0 if sequential else 0.2,
                "save_freq": 500,
                "merge_freq": 1000,
                "merge_alg": "ties",
                "objective_optimization": "only_label",
                "inner_params": ["language_model.model.layers[26].mlp.down_proj.weight"],
                "device": 0,
                "alg_name": "wise",
                "hidden_act": "silu",
                "force_adapter_output": True,
                "densities": 0.53,
                "weights": 1.0,
                "retrieve": False,
                "replay": False,
                "model_parallel": False,
                "use_chat_template": True,
            })

    return _H()


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_model(model_name: str = MODEL_NAME, device_map: str = "auto"):
    print("Instantiating Audio-Flamingo-3 model")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device_map,
    )
    model.eval()
    tok = processor.tokenizer
    return model, tok, processor


def _to_device(inputs: dict, model):
    """Move inputs to model device with correct dtype."""
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    return {
        k: (
            v.to(model_device, dtype=model_dtype)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
            else v.to(model_device) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }


def generate_audio_answer(model, processor, audio_path: str, question: str) -> str:
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": question}, {"type": "audio", "path": audio_path}]}
    ]
    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True)
    inputs = _to_device(inputs, model)
    with torch.no_grad():
        out_ids = model.generate(**inputs, **GEN_KWARGS)
        return processor.batch_decode(out_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)[0].strip()


def generate_text_answer(model, processor, question: str) -> str:
    conversation = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True)
    inputs = _to_device(inputs, model)
    with torch.no_grad():
        out_ids = model.generate(**inputs, **GEN_KWARGS)
        return processor.batch_decode(out_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# AF3 input builders (shared by UNKE and WISE)
# ---------------------------------------------------------------------------

def build_af3_audio_inputs(model, processor, data: Dict, device) -> dict:
    """Build AF3 audio inputs for a single edit example."""
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": data["question"]}, {"type": "audio", "path": data["audio_path"]}]}
    ]
    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True)
    model_dtype = next(model.parameters()).dtype
    return {
        k: (
            v.to(device, dtype=model_dtype)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
            else v.to(device) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }


def build_af3_text_inputs(model, processor, question: str, device) -> dict:
    """Build AF3 text-only inputs."""
    conversation = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True)
    model_dtype = next(model.parameters()).dtype
    return {
        k: (
            v.to(device, dtype=model_dtype)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
            else v.to(device) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }


# ---------------------------------------------------------------------------
# UNKE implementation (AF3-specific)
# ---------------------------------------------------------------------------

def _get_optimizer_params(layer_module, lr, weight_decay=0.01):
    no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    return [
        {
            "params": [p for n, p in layer_module.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": lr, "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in layer_module.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": lr, "weight_decay": 0.0,
        },
    ]


def compute_z_af3(model, processor, tok, data, layer, hparams):
    """Compute the target z vector for UNKE on Audio-Flamingo-3."""
    ln_f = nethook.get_module(model, hparams.ln_f_module)

    # Find lm_head weight
    lm_w_param = None
    for path in [f"{hparams.lm_head_module}.weight", "language_model.lm_head.weight"]:
        try:
            lm_w_param = nethook.get_parameter(model, path)
            break
        except LookupError:
            continue
    if lm_w_param is None:
        for n, p in model.named_parameters():
            if p.ndim == 2 and ("lm_head" in n or n.endswith("lm_head.weight")):
                lm_w_param = p
                break
    if lm_w_param is None:
        raise LookupError("Could not locate lm_head weight for Audio-Flamingo-3")
    lm_w = lm_w_param.T

    # Find lm_head bias
    lm_b = None
    for path in [f"{hparams.lm_head_module}.bias", "language_model.lm_head.bias"]:
        try:
            lm_b = nethook.get_parameter(model, path)
            break
        except LookupError:
            continue
    if lm_b is None:
        lm_b = next(model.parameters()).new_zeros(lm_w.shape[1])

    device = next(model.parameters()).device
    target_ids = tok(data["edited_answer"], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids.numel() > 0 and (target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id):
        target_ids = target_ids[1:]

    inputs = build_af3_audio_inputs(model, processor, data, device)

    # Append answer tokens (teacher forcing)
    new_input_ids = torch.cat([inputs["input_ids"], target_ids[:-1].unsqueeze(0)], dim=1)
    new_attention_mask = torch.cat([
        inputs["attention_mask"],
        torch.ones((inputs["attention_mask"].size(0), target_ids.size(0) - 1),
                    dtype=inputs["attention_mask"].dtype, device=device),
    ], dim=1)

    rewriting_targets = torch.full((new_input_ids.size(0), new_input_ids.size(1)), -100, device=device, dtype=torch.long)
    start = new_input_ids.size(1) - target_ids.size(0)
    rewriting_targets[:, start:] = target_ids
    lookup_idxs = [start]
    loss_layer = max(hparams.v_loss_layer, layer)

    down_proj_w = nethook.get_parameter(model, f"{hparams.layer_module_tmp.format(layer)}.mlp.down_proj.weight")
    hidden_size = down_proj_w.shape[0]
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device)
    target_init = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            out_tensor = cur_out[0]
            if out_tensor.dim() == 2:
                if target_init is None:
                    target_init = out_tensor[lookup_idxs[0]].detach().clone()
                out_tensor[lookup_idxs[0]] += delta
            else:
                if target_init is None:
                    target_init = out_tensor[0, lookup_idxs[0]].detach().clone()
                for i, idx in enumerate(lookup_idxs):
                    out_tensor[i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in tqdm(range(hparams.v_num_grad_steps), desc="compute_z_af3"):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[hparams.layer_module_tmp.format(loss_layer), hparams.layer_module_tmp.format(layer)],
            retain_input=False, retain_output=True, edit_output=edit_output_fn,
        ) as tr:
            run_inputs = dict(inputs)
            run_inputs["input_ids"] = new_input_ids
            run_inputs["attention_mask"] = new_attention_mask
            _ = model(**run_inputs)

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.dim() == 2:
            output = output.unsqueeze(0)
        if output.shape[1] != rewriting_targets.shape[1]:
            output = output.transpose(0, 1)

        log_probs = torch.log_softmax(ln_f(output) @ lm_w.to(output.device) + lm_b.to(output.device), dim=2)
        loss = torch.gather(
            log_probs, 2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        wd = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        total_loss = nll.mean() + wd.to(nll.device)

        if it == hparams.v_num_grad_steps - 1:
            break
        total_loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    del opt, delta
    nethook.set_requires_grad(False, model)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
    torch.cuda.empty_cache()
    return target


def compute_ks_af3_audio(model, processor, tok, batch_data, hparams, layer):
    """Compute current z vectors (before editing) for AF3 with audio inputs."""
    device = next(model.parameters()).device
    zs, idxs = [], []
    for d in batch_data:
        inputs = build_af3_audio_inputs(model, processor, d, device)
        with torch.no_grad():
            with nethook.Trace(
                module=model, layer=hparams.layer_module_tmp.format(layer),
                retain_input=True, retain_output=True, detach=True, clone=True,
            ) as tr:
                _ = model(**inputs)
                out = tr.output
        out = out[0] if isinstance(out, tuple) else out
        last_idx = int(inputs["attention_mask"].sum().item()) - 1
        zs.append(out[0, last_idx])
        idxs.append(last_idx)
    return torch.stack(zs, dim=0), idxs


def apply_unke_af3(model, processor, tok, hparams, batch_data, ex_data, audio_ex_data=None):
    """Apply UNKE algorithm to Audio-Flamingo-3 model."""
    from editing_method.algo.unke.unke_main import get_qwen2_causal_mask

    # Backup target-layer weights
    preserve_params = []
    for name, _ in model.named_parameters():
        for layer_idx in hparams.layers:
            if f".layers.{layer_idx}." in name:
                preserve_params.append(name)
                break
    weights = {p: nethook.get_parameter(model, p) for p in preserve_params}
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    device = next(model.parameters()).device
    z_layer = hparams.layers[-1]
    zs = torch.stack([compute_z_af3(model, processor, tok, d, z_layer, hparams) for d in batch_data])

    for i, layer in enumerate(hparams.layers):
        with torch.no_grad():
            with nethook.Trace(
                module=model, layer=hparams.layer_module_tmp.format(layer),
                retain_input=True, retain_output=True, detach=True, clone=True,
            ) as tr:
                inputs = build_af3_audio_inputs(model, processor, batch_data[0], device)
                _ = model(**inputs)
                layer_in_ks = tr.input
                layer_out_ks = tr.output
        layer_out_ks = layer_out_ks[0] if isinstance(layer_out_ks, tuple) else layer_out_ks
        if layer_in_ks.dim() == 2:
            layer_in_ks = layer_in_ks.unsqueeze(0)
        if layer_out_ks.dim() == 2:
            layer_out_ks = layer_out_ks.unsqueeze(0)

        cur_zs, idxs = compute_ks_af3_audio(model, processor, tok, batch_data, hparams, z_layer)
        targets = zs - cur_zs

        # Text preservation stats
        text_preserve = []
        if ex_data:
            sample_texts = ex_data if len(ex_data) <= 4 else random.sample(ex_data, 4)
            for text_q in sample_texts:
                text_inputs = build_af3_text_inputs(model, processor, text_q, device)
                with torch.no_grad():
                    with nethook.Trace(
                        module=model, layer=hparams.layer_module_tmp.format(layer),
                        retain_input=True, retain_output=True, detach=True, clone=True,
                    ) as tr_t:
                        _ = model(**text_inputs)
                t_in, t_out = tr_t.input, tr_t.output
                t_out = t_out[0] if isinstance(t_out, tuple) else t_out
                if t_in.dim() == 2:
                    t_in = t_in.unsqueeze(0)
                if t_out.dim() == 2:
                    t_out = t_out.unsqueeze(0)
                text_preserve.append((t_in, t_out, text_inputs.get("attention_mask")))

        # Audio preservation stats
        audio_preserve = []
        if audio_ex_data:
            sample_items = audio_ex_data if len(audio_ex_data) <= 4 else random.sample(audio_ex_data, 4)
            for item in sample_items:
                inputs_i = build_af3_audio_inputs(model, processor, item, device)
                with torch.no_grad():
                    with nethook.Trace(
                        module=model, layer=hparams.layer_module_tmp.format(layer),
                        retain_input=True, retain_output=True, detach=True, clone=True,
                    ) as tr_a:
                        _ = model(**inputs_i)
                a_in, a_out = tr_a.input, tr_a.output
                a_out = a_out[0] if isinstance(a_out, tuple) else a_out
                if a_in.dim() == 2:
                    a_in = a_in.unsqueeze(0)
                if a_out.dim() == 2:
                    a_out = a_out.unsqueeze(0)
                audio_preserve.append((a_in, a_out, inputs_i.get("attention_mask")))

        resid = targets / (len(hparams.layers) - i)
        criterion = torch.nn.MSELoss()
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        for _, m in _layer.named_parameters():
            m.requires_grad = True
        optimizer = torch.optim.AdamW(
            _get_optimizer_params(_layer, hparams.lr), lr=hparams.lr, eps=1e-8, betas=(0.9, 0.999),
        )

        for j in range(len(idxs)):
            layer_out_ks[j, idxs[j]] += resid[j]

        input_causal_mask, input_position_ids = get_qwen2_causal_mask(
            layer_in_ks, torch.ones_like(layer_in_ks[:, :, 0]),
        )

        for step in tqdm(range(hparams.optim_num_step)):
            optimizer.zero_grad()

            # Text preservation loss
            loss_text = 0.0
            if text_preserve:
                for t_in, t_out, t_attn in text_preserve:
                    t_mask, t_pos = get_qwen2_causal_mask(t_in, t_attn)
                    t_emb = model.language_model.model.embed_tokens(t_pos)
                    t_pe = model.language_model.model.rotary_emb(t_emb, t_pos)
                    out = _layer(t_in, attention_mask=t_mask, position_embeddings=t_pe)[0]
                    if out.dim() == 2:
                        out = out.unsqueeze(0)
                    loss_text = loss_text + criterion(out, t_out)
                loss_text = loss_text / float(len(text_preserve))

            # Edit context loss
            in_emb = model.language_model.model.embed_tokens(input_position_ids)
            in_pe = model.language_model.model.rotary_emb(in_emb, input_position_ids)
            out_edit = _layer(layer_in_ks, attention_mask=input_causal_mask, position_embeddings=in_pe)[0]
            if out_edit.dim() == 2:
                out_edit = out_edit.unsqueeze(0)
            loss_edit = criterion(out_edit, layer_out_ks)

            # Audio preservation loss
            loss_audio = 0.0
            if audio_preserve:
                for a_in, a_out, a_attn in audio_preserve:
                    a_mask, a_pos = get_qwen2_causal_mask(a_in, a_attn)
                    a_emb = model.language_model.model.embed_tokens(a_pos)
                    a_pe = model.language_model.model.rotary_emb(a_emb, a_pos)
                    out_a = _layer(a_in, attention_mask=a_mask, position_embeddings=a_pe)[0]
                    if out_a.dim() == 2:
                        out_a = out_a.unsqueeze(0)
                    loss_audio = loss_audio + criterion(out_a, a_out)
                loss_audio = loss_audio / float(len(audio_preserve))

            loss = loss_edit
            if text_preserve:
                loss = loss + loss_text
            if audio_preserve:
                loss = loss + loss_audio
            loss.backward(retain_graph=True)
            optimizer.step()

        for _, m in _layer.named_parameters():
            if m.grad is not None:
                m.grad.data.zero_()
            m.requires_grad = False
        del optimizer
        torch.cuda.empty_cache()

    return weights_copy


# ---------------------------------------------------------------------------
# WISE tokenization (AF3-specific)
# ---------------------------------------------------------------------------

def _build_af3_audio_batch(processor, requests, device, model_dtype, include_labels=True):
    """Build audio batch for AF3 WISE editing."""
    if not requests:
        return None
    tokenizer = processor.tokenizer

    question_inputs_list, full_inputs_list = [], []
    for req in requests:
        conv_user = [
            {"role": "user", "content": [
                {"type": "text", "text": req.get("question", "")},
                {"type": "audio", "path": req.get("audio_path")},
            ]}
        ]
        q_inputs = processor.apply_chat_template(conv_user, tokenize=True, add_generation_prompt=True, return_dict=True)
        question_inputs_list.append(q_inputs)

        answer = (req.get("edited_answer", "") or "").strip()
        if include_labels and answer:
            conv_full = conv_user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
            f_inputs = processor.apply_chat_template(conv_full, tokenize=True, add_generation_prompt=False, return_dict=True)
        else:
            f_inputs = q_inputs
        full_inputs_list.append(f_inputs)

    full_inputs = full_inputs_list[0]
    question_inputs = question_inputs_list[0]

    processed = {}
    for k, v in full_inputs.items():
        if isinstance(v, torch.Tensor):
            processed[k] = v.to(device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(device)
        else:
            processed[k] = v

    if include_labels:
        prefix_len = question_inputs["input_ids"].size(1)
        labels = processed["input_ids"].clone()
        labels[:, :prefix_len] = -100
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        processed["labels"] = labels

    return processed


def _build_text_locality_inputs(tokenizer, prompts, device):
    if not prompts:
        return None
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    tokenized["labels"] = torch.full_like(tokenized["input_ids"], -100, dtype=torch.long)
    return {k: v.to(device) for k, v in tokenized.items()}


def multimodal_tokenize_af3(requests, processor, model, device, text_locality=None, audio_locality=None):
    """Prepare edit and locality batches for AF3 WISE editing."""
    if not requests:
        raise ValueError("At least one edit request is required.")
    model_dtype = next(model.parameters()).dtype
    tokenizer = processor.tokenizer

    edit_batch = _build_af3_audio_batch(processor, requests, device, model_dtype, include_labels=True)

    text_locality_inputs = None
    if text_locality:
        prompts = [item.get("question", "").strip() for item in text_locality if item.get("question", "").strip()]
        if prompts:
            text_locality_inputs = _build_text_locality_inputs(tokenizer, prompts, device)

    audio_locality_inputs = None
    if audio_locality:
        audio_locality_inputs = _build_af3_audio_batch(processor, audio_locality, device, model_dtype, include_labels=False)

    return {
        "edit_batch": edit_batch,
        "text_locality_inputs": text_locality_inputs,
        "audio_locality_inputs": audio_locality_inputs,
    }


def apply_wise_to_af3_model(model, processor, requests, hparams, text_locality=None, audio_locality=None):
    """Apply WISE algorithm to Audio-Flamingo-3 model (single edit)."""
    from editing_method.algo.wise.WISE import WISEMultimodal

    device = f"cuda:{hparams.device}"
    model.to(device)
    editor = WISEMultimodal(model=model, config=hparams, device=device)

    print("Executing WISE algorithm for the update: ")
    for req in requests:
        print(f"[{req['question']}] -> [{req['edited_answer']}]")

    payload = multimodal_tokenize_af3(
        requests=requests, processor=processor, model=model, device=device,
        text_locality=text_locality, audio_locality=audio_locality,
    )
    editor.edit(
        config=hparams,
        edit_batch=payload["edit_batch"],
        text_locality_inputs=payload.get("text_locality_inputs"),
        audio_locality_inputs=payload.get("audio_locality_inputs"),
    )
    return editor, editor.reset_layer


# ---------------------------------------------------------------------------
# Run: UNKE single edits
# ---------------------------------------------------------------------------

def run_unke(args, model, tok, processor, params, text_preserve, audio_preserve, meta_items):
    text_formatted = [item["question"] for item in text_preserve]
    audio_ex_data = [{"question": item["question"], "audio_path": item["audio_path"]} for item in audio_preserve]

    def apply_edit(item, audio_path):
        batch_data = [{
            "question": item.get("reliability_question", ""),
            "audio_path": audio_path,
            "edited_answer": item.get("edited_answer", ""),
        }]
        sampled_text = random.sample(text_formatted, min(len(text_formatted), params.ex_data_num))
        sampled_audio = (
            random.sample(audio_ex_data, min(len(audio_ex_data), args.audio_ex_data_num))
            if audio_ex_data else None
        )
        return apply_unke_af3(model, processor, tok, params, batch_data, sampled_text, sampled_audio)

    def restore(weights_copy):
        with torch.no_grad():
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(device=param.device, dtype=param.dtype)

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "unke", "category": args.category},
        prefix="af3_unke",
    )

    gen_text = lambda q: generate_text_answer(model, processor, q)
    gen_audio_factory = lambda item: (lambda ap, q: generate_audio_answer(model, processor, ap, q))

    run_single_edits(
        meta_items, args.dataset_size_limit, args.category,
        gen_audio_factory, gen_text, apply_edit, restore, model, writer,
        edit_only=args.edit_only,
    )


# ---------------------------------------------------------------------------
# Run: WISE single edits
# ---------------------------------------------------------------------------

def run_wise_single(args, model, tok, processor, params, text_preserve, audio_preserve, meta_items):
    def apply_edit(item, audio_path):
        batch_data = [{
            "question": item.get("reliability_question", ""),
            "edited_answer": item.get("edited_answer", ""),
            "audio_path": audio_path,
        }]
        _, weights_copy = apply_wise_to_af3_model(
            model=model, processor=processor, requests=batch_data, hparams=params,
            text_locality=random.sample(text_preserve, min(len(text_preserve), 10)),
            audio_locality=random.sample(audio_preserve, min(len(audio_preserve), 10)),
        )
        return weights_copy

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "wise", "category": args.category},
        prefix="af3_wise",
    )

    gen_text = lambda q: generate_text_answer(model, processor, q)
    gen_audio_factory = lambda item: (lambda ap, q: generate_audio_answer(model, processor, ap, q))

    run_single_edits(
        meta_items, args.dataset_size_limit, args.category,
        gen_audio_factory, gen_text, apply_edit,
        lambda wc: restore_model_weights(model, wc),
        model, writer,
    )


# ---------------------------------------------------------------------------
# Run: WISE sequential edits
# ---------------------------------------------------------------------------

def run_wise_sequential(args, model, processor, params, text_preserve, audio_preserve, groups_data):
    from editing_method.algo.wise.WISE import WISEMultimodal

    device = f"cuda:{params.device}"

    main_memory_weights = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                main_memory_weights[name] = param.detach().clone()

    def setup_group():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in main_memory_weights:
                    param.copy_(main_memory_weights[name])
        cleanup_cuda(model)
        editor = WISEMultimodal(model=model, config=params, device=device)
        adapter = editor.get_adapter_layer()
        adapter.used_mask = None
        return editor, adapter

    def apply_edit(editor, item, audio_path):
        batch_data = [{
            "question": item.get("reliability_question", ""),
            "edited_answer": item.get("edited_answer", ""),
            "audio_path": audio_path,
        }]
        text_batch = (
            random.sample(text_preserve, min(len(text_preserve), args.text_ex_data_num))
            if text_preserve else []
        )
        audio_batch = (
            random.sample(audio_preserve, min(len(audio_preserve), args.audio_ex_data_num))
            if audio_preserve else []
        )
        payload = multimodal_tokenize_af3(
            requests=batch_data, processor=processor, model=model, device=device,
            text_locality=text_batch, audio_locality=audio_batch,
        )
        editor.edit(
            config=params,
            edit_batch=payload["edit_batch"],
            text_locality_inputs=payload.get("text_locality_inputs"),
            audio_locality_inputs=payload.get("audio_locality_inputs"),
        )

    def reset_group(editor):
        if editor is not None:
            editor.reset_layer()

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "wise", "sequential_editing": True, "category": args.category},
        prefix="af3_wise_seq",
    )

    gen_audio = lambda ap, q: generate_audio_answer(model, processor, ap, q)
    gen_text = lambda q: generate_text_answer(model, processor, q)

    run_sequential_edits(
        groups_data, args.dataset_size_limit, gen_audio, gen_text,
        setup_group, apply_edit, reset_group, model, writer,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Knowledge editing for Audio-Flamingo-3")
    parser.add_argument("--algorithm", choices=["unke", "wise"], required=True)
    parser.add_argument("--mode", choices=["single", "sequential"], default="single")
    parser.add_argument("--category", default="Language")
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dataset-size-limit", type=int, default=300)
    parser.add_argument("--audio-ex-data-num", type=int, default=5)
    parser.add_argument("--text-ex-data-num", type=int, default=10)
    parser.add_argument("--edit-only", action="store_true")
    args = parser.parse_args()

    if args.metadata_file is None:
        if args.mode == "sequential":
            args.metadata_file = "lalm-knowledge-editing/metadata/test/seq_update.json"
            args.category = "sequential_edits"
        else:
            args.metadata_file = (
                f"lalm-knowledge-editing/metadata/test/{args.category}_transcriptions_no_label.json"
            )

    set_seed()

    # UNKE uses device_map="auto", WISE uses explicit device placement
    if args.algorithm == "unke":
        model, tok, processor = load_model(args.model_name, device_map="auto")
    else:
        model, tok, processor = load_model(args.model_name, device_map=None)
        params = make_wise_hparams(args.category, sequential=(args.mode == "sequential"))
        model = model.to(f"cuda:{params.device}")

    ex_datas = load_preservation_data_raw()
    text_preserve = build_text_preserve(ex_datas["text"])
    audio_preserve = build_audio_preserve(ex_datas["audio"], exclude_category=args.category)

    if args.algorithm == "unke":
        params = make_unke_hparams(args.category)
        run_unke(args, model, tok, processor, params, text_preserve, audio_preserve, load_metadata(args.metadata_file))
    elif args.mode == "sequential":
        groups_data = load_groups_data(args.metadata_file)
        run_wise_sequential(args, model, processor, params, text_preserve, audio_preserve, groups_data)
    else:
        params = make_wise_hparams(args.category, sequential=False)
        run_wise_single(args, model, tok, processor, params, text_preserve, audio_preserve, load_metadata(args.metadata_file))


if __name__ == "__main__":
    main()
