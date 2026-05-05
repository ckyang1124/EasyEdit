#!/usr/bin/env python3
"""Knowledge editing for DeSTA2.5-Audio model.

Supports UNKE and WISE algorithms in single-edit and sequential modes.

Usage:
    python edit_desta.py --algorithm unke --category Language
    python edit_desta.py --algorithm wise --category Language
    python edit_desta.py --algorithm wise --mode sequential
"""
import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from DeSTA25_Audio.desta.models.modeling_desta25 import DeSTA25AudioModel
from editing_method.dataset_classes.templates import TEMPLATE_DICT
from editing_method.util import nethook

from edit_common import (
    set_seed,
    resolve_audio_path,
    load_preservation_data_raw,
    build_text_preserve,
    build_audio_preserve,
    build_audio_preserve_only,
    load_metadata,
    load_groups_data,
    run_single_edits,
    run_sequential_edits,
    restore_model_weights,
    cleanup_cuda,
    ResultsWriter,
)

HF_TOKEN = ""
MODEL_NAME = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"


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
                "layers": [15],
                "clamp_norm_factor": 4,
                "layer_selection": "all",
                "fact_token": "last",
                "lr": 2e-4,
                "v_num_grad_steps": 25,
                "v_lr": 5e-1,
                "v_loss_layer": 31,
                "v_weight_decay": 1e-3,
                "optim_num_step": 50,
                "ex_data_num": 30,
                "rewrite_module_tmp": "llm_model.model.layers.{}.mlp.down_proj",
                "layer_module_tmp": "llm_model.model.layers.{}",
                "mlp_module_tmp": "llm_model.model.layers.{}.mlp",
                "attn_module_tmp": "llm_model.model.layers.{}.self_attn",
                "ln_f_module": "llm_model.model.norm",
                "lm_head_module": "llm_model.lm_head",
                "arg_note": "desta25-multimodal",
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
                "edit_lr": 0.1,
                "n_iter": 50,
                "norm_constraint": 1.0,
                "alpha": 2.0,
                "beta": 20.0,
                "gamma": 10.0,
                "act_ratio": 0.88,
                "save_freq": 500,
                "merge_freq": 1000,
                "merge_alg": "ties",
                "objective_optimization": "only_label",
                "inner_params": ["llm_model.model.layers[29].mlp.down_proj.weight"],
                "device": 0,
                "alg_name": "wise",
                "hidden_act": "silu",
                "force_adapter_output": False,
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

def load_model(model_name: str = MODEL_NAME):
    print("Instantiating DeSTA model")
    model = DeSTA25AudioModel.from_pretrained(model_name, token=HF_TOKEN)
    model = model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    tok = AutoTokenizer.from_pretrained(
        model.config.llm_model_id, cache_dir=os.getenv("HF_HOME")
    )
    tok.padding_side = "left"
    tok.pad_token_id = tok.eos_token_id
    tok.add_tokens(["<|AUDIO|>"])
    tok.add_tokens(["<|reserved_special_token_87|>"])
    return model, tok


def generate_audio_answer(model, audio_path: str, question: str, transcription: str = "") -> str:
    messages = [
        {
            "role": "user",
            "content": f"<|AUDIO|>\n{question}",
            "audios": [{"audio": audio_path, "text": transcription}],
        }
    ]
    out = model.generate(messages=messages, do_sample=False, top_p=1.0, temperature=1.0, max_new_tokens=128)
    return out.text[0] if isinstance(out.text, list) else out.text


def generate_text_answer(model, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    out = model.generate(messages=messages, do_sample=False, top_p=1.0, temperature=1.0, max_new_tokens=128)
    return out.text[0] if isinstance(out.text, list) else out.text


# ---------------------------------------------------------------------------
# Run: UNKE single edits
# ---------------------------------------------------------------------------

def run_unke(args, model, tok, params, text_preserve, audio_preserve, meta_items):
    from editing_method.algo.unke.unke_main import apply_unke_to_model

    template = TEMPLATE_DICT["DeSTA25-Audio-Llama-3.1-8B"]
    text_formatted = [template.wo_answer(item["question"]) for item in text_preserve]

    def apply_edit(item, audio_path):
        batch_data = [{
            "question": item.get("reliability_question", ""),
            "edited_answer": item.get("edited_answer", ""),
            "audio_path": audio_path,
            "transcription": item.get("transcription", ""),
        }]
        return apply_unke_to_model(
            model, tok, params, batch_data,
            ex_data=random.sample(text_formatted, min(len(text_formatted), params.ex_data_num)),
            audio_ex_data=random.sample(audio_preserve, min(len(audio_preserve), args.audio_ex_data_num)),
        )

    def restore(weights_copy):
        with torch.no_grad():
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(device=param.device, dtype=param.dtype)

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "unke", "category": args.category},
        prefix="desta_unke",
    )

    gen_text = lambda q: generate_text_answer(model, q)
    gen_audio_factory = lambda item: (
        lambda ap, q: generate_audio_answer(model, ap, q, item.get("transcription", ""))
    )

    run_single_edits(
        meta_items, args.dataset_size_limit, args.category,
        gen_audio_factory, gen_text, apply_edit, restore, model, writer,
        edit_only=args.edit_only,
    )


# ---------------------------------------------------------------------------
# Run: WISE single edits
# ---------------------------------------------------------------------------

def run_wise_single(args, model, tok, params, text_preserve, audio_preserve, meta_items):
    from editing_method.algo.wise.wise_main import apply_wise_to_multimodal_model

    def apply_edit(item, audio_path):
        batch_data = [{
            "question": item.get("reliability_question", ""),
            "edited_answer": item.get("edited_answer", ""),
            "audio_path": audio_path,
            "transcription": item.get("transcription", ""),
        }]
        _, weights_copy = apply_wise_to_multimodal_model(
            model, tok, batch_data, params,
            text_locality=random.sample(text_preserve, min(len(text_preserve), 10)),
            audio_locality=random.sample(audio_preserve, min(len(audio_preserve), 10)),
        )
        return weights_copy

    writer = ResultsWriter(
        args.model_name, args.metadata_file, args.dataset_size_limit,
        extra_config={"alg_name": "wise", "category": args.category},
        prefix="desta_wise",
    )

    gen_text = lambda q: generate_text_answer(model, q)
    gen_audio_factory = lambda item: (
        lambda ap, q: generate_audio_answer(model, ap, q, item.get("transcription", ""))
    )

    run_single_edits(
        meta_items, args.dataset_size_limit, args.category,
        gen_audio_factory, gen_text, apply_edit,
        lambda wc: restore_model_weights(model, wc),
        model, writer,
    )


# ---------------------------------------------------------------------------
# Run: WISE sequential edits
# ---------------------------------------------------------------------------

def run_wise_sequential(args, model, tok, params, text_preserve, audio_preserve, groups_data):
    from editing_method.algo.wise.WISE import WISEMultimodal
    from editing_method.algo.wise.utils import multimodal_tokenize

    device = f"cuda:{params.device}"

    # Snapshot the untouched weights that serve as WISE's main memory
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
            "transcription": item.get("transcription", ""),
        }]
        text_batch = (
            random.sample(text_preserve, min(len(text_preserve), args.text_ex_data_num))
            if text_preserve else []
        )
        audio_batch = (
            random.sample(audio_preserve, min(len(audio_preserve), args.audio_ex_data_num))
            if audio_preserve else []
        )
        payload = multimodal_tokenize(
            requests=batch_data, tokenizer=tok, model=model, device=device,
            hparams=params, text_locality=text_batch, audio_locality=audio_batch,
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
        prefix="desta_wise_seq",
    )

    gen_audio = lambda ap, q: generate_audio_answer(model, ap, q)
    gen_text = lambda q: generate_text_answer(model, q)

    run_sequential_edits(
        groups_data, args.dataset_size_limit, gen_audio, gen_text,
        setup_group, apply_edit, reset_group, model, writer,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Knowledge editing for DeSTA2.5-Audio")
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

    # Default metadata file
    if args.metadata_file is None:
        if args.mode == "sequential":
            args.metadata_file = "lalm-knowledge-editing/metadata/test/seq_update.json"
            args.category = "sequential_edits"
        else:
            args.metadata_file = (
                f"lalm-knowledge-editing/metadata/test/{args.category}_transcriptions_no_label.json"
            )

    set_seed()
    model, tok = load_model(args.model_name)

    # Preservation data
    ex_datas = load_preservation_data_raw()
    text_preserve = build_text_preserve(ex_datas["text"])
    if args.mode == "sequential":
        audio_preserve = build_audio_preserve_only(ex_datas["audio"], "Dynamic_Superb")
    else:
        audio_preserve = build_audio_preserve(ex_datas["audio"], exclude_category=args.category)

    # Hyperparameters
    params = make_unke_hparams(args.category) if args.algorithm == "unke" else make_wise_hparams(args.category)

    # Dispatch
    if args.mode == "sequential":
        groups_data = load_groups_data(args.metadata_file)
        run_wise_sequential(args, model, tok, params, text_preserve, audio_preserve, groups_data)
    elif args.algorithm == "unke":
        run_unke(args, model, tok, params, text_preserve, audio_preserve, load_metadata(args.metadata_file))
    else:
        run_wise_single(args, model, tok, params, text_preserve, audio_preserve, load_metadata(args.metadata_file))


if __name__ == "__main__":
    main()
