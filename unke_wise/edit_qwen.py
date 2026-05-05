#!/usr/bin/env python3
"""Knowledge editing for Qwen2-Audio model.

Supports UNKE and WISE algorithms in single-edit and sequential modes.

Usage:
    python edit_qwen.py --algorithm unke --category Language
    python edit_qwen.py --algorithm wise --category Language
    python edit_qwen.py --algorithm wise --mode sequential
"""
import argparse
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, Qwen2AudioForConditionalGeneration

from editing_method.util import nethook
from editing_method.dataset_classes.templates import TEMPLATE_DICT

from edit_common import (
    set_seed,
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

try:
    import librosa  # type: ignore
except ImportError:
    librosa = None

HF_TOKEN = ""
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

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
                "layers": [27],
                "clamp_norm_factor": 4,
                "layer_selection": "all",
                "fact_token": "last",
                "ex_data_num": 20,
                "lr": 2e-4,
                "v_num_grad_steps": 25,
                "v_lr": 5e-1,
                "v_loss_layer": 31,
                "v_weight_decay": 1e-3,
                "optim_num_step": 50,
                "rewrite_module_tmp": "language_model.model.layers.{}.mlp.down_proj",
                "layer_module_tmp": "language_model.model.layers.{}",
                "mlp_module_tmp": "language_model.model.layers.{}.mlp",
                "attn_module_tmp": "language_model.model.layers.{}.self_attn",
                "ln_f_module": "language_model.model.norm",
                "lm_head_module": "language_model.lm_head",
            })

    return _H()


def make_wise_hparams(category: str):
    from editing_method.algo.wise.wise_hparams import WISEHyperParams

    class _H(WISEHyperParams):
        def __init__(self):
            super().__init__(**{
                "model_name": MODEL_NAME,
                "ds_name": category,
                "mask_ratio": 0.2,
                "edit_lr": 1.0,
                "n_iter": 70,
                "norm_constraint": 1.0,
                "alpha": 5.0,
                "beta": 20.0,
                "gamma": 10.0,
                "act_ratio": 0.88,
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

def _load_audio(audio_path: str, target_sr: int):
    if librosa is None:
        raise ImportError("librosa is required. Install with: pip install librosa")
    wav, sr = librosa.load(audio_path, sr=target_sr)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def load_model(model_name: str = MODEL_NAME):
    print("Instantiating Qwen2-Audio model")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, token=HF_TOKEN,
    ).to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
    return model, tok, processor


def generate_audio_answer(model, processor, audio_path: str, question: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio = _load_audio(audio_path, processor.feature_extractor.sampling_rate)
    inputs = processor(
        text=[prompt], audio=[audio], return_tensors="pt", padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (
            v.to("cuda", dtype=model_dtype)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
            else v.to("cuda") if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }
    with torch.no_grad():
        out_ids = model.generate(**inputs, **GEN_KWARGS)
        gen_only = out_ids[:, inputs["input_ids"].size(1):]
        return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def generate_text_answer(model, processor, question: str) -> str:
    conversation = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, **GEN_KWARGS)
        gen_only = out_ids[:, inputs["input_ids"].size(1):]
        return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


# ---------------------------------------------------------------------------
# UNKE implementation (Qwen-specific)
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


def apply_unke_qwen(model, processor, tok, hparams, batch_data, ex_data, audio_ex_data=None):
    """Apply UNKE algorithm to Qwen2-Audio model."""
    from editing_method.algo.unke.compute_z import compute_z_qwen, compute_ks_qwen_audio, build_qwen_audio_inputs
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

    # Compute target z vectors
    zs = torch.stack([compute_z_qwen(model, processor, tok, d, z_layer, hparams) for d in batch_data])

    for i, layer in enumerate(hparams.layers):
        # Trace layer IO on edit context
        with torch.no_grad():
            with nethook.Trace(
                module=model, layer=hparams.layer_module_tmp.format(layer),
                retain_input=True, retain_output=True, detach=True, clone=True,
            ) as tr:
                inputs, _ = build_qwen_audio_inputs(model, processor, tok, batch_data[0], device)
                _ = model(**inputs)
                layer_in_ks = tr.input
                layer_out_ks = tr.output
        layer_out_ks = layer_out_ks[0] if isinstance(layer_out_ks, tuple) else layer_out_ks

        cur_zs, idxs = compute_ks_qwen_audio(model, processor, tok, batch_data, hparams, z_layer)
        targets = zs - cur_zs

        # Text preservation stats
        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            with nethook.Trace(
                module=model, layer=hparams.layer_module_tmp.format(layer),
                retain_input=True, retain_output=True, detach=True, clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if isinstance(stat_out, tuple) else stat_out

        # Audio preservation stats
        audio_preserve = []
        if audio_ex_data:
            sample_items = audio_ex_data if len(audio_ex_data) <= 4 else random.sample(audio_ex_data, 4)
            for item in sample_items:
                inputs_i, _ = build_qwen_audio_inputs(model, processor, tok, item, device)
                with torch.no_grad():
                    with nethook.Trace(
                        module=model, layer=hparams.layer_module_tmp.format(layer),
                        retain_input=True, retain_output=True, detach=True, clone=True,
                    ) as tr_a:
                        _ = model(**inputs_i)
                        a_in = tr_a.input
                        a_out = tr_a.output
                a_out = a_out[0] if isinstance(a_out, tuple) else a_out
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
        ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(stat_in, ex_tok["attention_mask"])

        for step in tqdm(range(hparams.optim_num_step)):
            optimizer.zero_grad()
            # Text preservation loss
            ex_embeds = model.language_model.model.embed_tokens(ex_position_ids)
            ex_pos_emb = model.language_model.model.rotary_emb(ex_embeds, ex_position_ids)
            loss1 = criterion(
                _layer(stat_in, attention_mask=ex_causal_mask, position_embeddings=ex_pos_emb)[0],
                stat_out,
            )
            # Edit context loss
            in_embeds = model.language_model.model.embed_tokens(input_position_ids)
            in_pos_emb = model.language_model.model.rotary_emb(in_embeds, input_position_ids)
            loss2 = criterion(
                _layer(layer_in_ks, attention_mask=input_causal_mask, position_embeddings=in_pos_emb)[0],
                layer_out_ks,
            )
            # Audio preservation loss
            loss = loss1 + loss2
            if audio_preserve:
                loss_audio = 0.0
                for a_in, a_out, a_attn in audio_preserve:
                    a_mask, a_pos_ids = get_qwen2_causal_mask(a_in, a_attn)
                    a_embeds = model.language_model.model.embed_tokens(a_pos_ids)
                    a_pos_emb = model.language_model.model.rotary_emb(a_embeds, a_pos_ids)
                    loss_audio = loss_audio + criterion(
                        _layer(a_in, attention_mask=a_mask, position_embeddings=a_pos_emb)[0], a_out,
                    )
                loss = loss + loss_audio / float(len(audio_preserve))
            loss.backward(retain_graph=True)
            optimizer.step()

    return weights_copy


# ---------------------------------------------------------------------------
# Run: UNKE single edits
# ---------------------------------------------------------------------------

def run_unke(args, model, tok, processor, params, text_preserve, audio_preserve, meta_items):
    template = TEMPLATE_DICT["Qwen2-7B-Instruct"]
    text_formatted = [template.wo_answer(item["question"]) for item in text_preserve]

    audio_ex_data = [
        {"question": item["question"], "audio_path": item["audio_path"]}
        for item in audio_preserve
    ]

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
        return apply_unke_qwen(model, processor, tok, params, batch_data, sampled_text, sampled_audio)

    def restore(weights_copy):
        with torch.no_grad():
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(device=param.device, dtype=param.dtype)

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "unke", "category": args.category},
        prefix="qwen_unke",
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
    from editing_method.algo.wise.wise_main import apply_wise_to_qwen_model

    def apply_edit(item, audio_path):
        batch_data = [{
            "question": item.get("reliability_question", ""),
            "edited_answer": item.get("edited_answer", ""),
            "audio_path": audio_path,
        }]
        _, weights_copy = apply_wise_to_qwen_model(
            model=model, processor=processor, requests=batch_data, hparams=params,
            text_locality=random.sample(text_preserve, min(len(text_preserve), 10)),
            audio_locality=random.sample(audio_preserve, min(len(audio_preserve), 10)),
        )
        return weights_copy

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "wise", "category": args.category},
        prefix="qwen_wise",
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
    from editing_method.algo.wise.utils import multimodal_tokenize_qwen

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
        payload = multimodal_tokenize_qwen(
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
        prefix="qwen_wise_seq",
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
    parser = argparse.ArgumentParser(description="Knowledge editing for Qwen2-Audio")
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
    model, tok, processor = load_model(args.model_name)

    ex_datas = load_preservation_data_raw()
    text_preserve = build_text_preserve(ex_datas["text"])
    audio_preserve = build_audio_preserve(ex_datas["audio"], exclude_category=args.category)

    params = make_unke_hparams(args.category) if args.algorithm == "unke" else make_wise_hparams(args.category)

    if args.mode == "sequential":
        groups_data = load_groups_data(args.metadata_file)
        run_wise_sequential(args, model, processor, params, text_preserve, audio_preserve, groups_data)
    elif args.algorithm == "unke":
        run_unke(args, model, tok, processor, params, text_preserve, audio_preserve, load_metadata(args.metadata_file))
    else:
        run_wise_single(args, model, tok, processor, params, text_preserve, audio_preserve, load_metadata(args.metadata_file))


if __name__ == "__main__":
    main()
