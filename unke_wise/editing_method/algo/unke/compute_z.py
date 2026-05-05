from typing import Dict, List, Tuple
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .unke_hparams import unkeHyperParams
from ...util import nethook
from tqdm import tqdm
from DeSTA25_Audio.desta.models.modeling_desta25 import DeSTA25AudioModel, _prepare_audio_context_and_start_positions
from DeSTA25_Audio.desta.utils.audio import AudioSegment
from transformers import AutoProcessor
from transformers import Qwen2AudioForConditionalGeneration


def _ensure_desta_resources(model: DeSTA25AudioModel):
    if not hasattr(model, "tokenizer") or not hasattr(model, "processor"):
        model._setup_generation()
    return model.tokenizer, model.processor


def build_desta_audio_context_and_features(model: DeSTA25AudioModel, data, device):
    # 1) Ensure tokenizer/processor are ready
    tok, _ = _ensure_desta_resources(model)

    # 2) Prepare transcription sizes
    transcriptions = [data.get("transcription", "")]  # or "" if None
    transcription_size_list = [len(tok.tokenize(t, add_special_tokens=False)) for t in transcriptions]
    audio_size_list = [model.config.prompt_size] * len(transcriptions)

    # 3) Build one-turn chat with the audio locator
    messages = [
        # {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": f"<|AUDIO|>\n{data['question']}",
         "audios": [{"audio": data["audio_path"], "text": None}]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text.replace(model.config.audio_locator, f"<start_audio>{model.config.audio_locator}<end_audio>")

    # 4) Convert to token list, replace <|AUDIO|> with placeholder span, and record start positions
    token_list = tok.tokenize(text)
    token_list, start_positions = _prepare_audio_context_and_start_positions(
        token_list=token_list,
        audio_locator=model.config.audio_locator,
        audio_size_list=audio_size_list.copy(),
        transcription_size_list=transcription_size_list.copy(),
        placeholder_token=model.config.placeholder_token,
    )
    audio_context = tok.convert_tokens_to_string(token_list)

    # 5) Tokenize to ids
    ctx = tok(
        [audio_context],
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_length=True,
        add_special_tokens=False,
    ).to(device)
    # Shift start positions by left padding
    pad = (ctx["length"][0] - ctx["attention_mask"][0].sum()).item()
    batch_start_positions = [(0, s + pad) for s in start_positions]

    # 6) Load and featurize audio
    feature = AudioSegment.from_file(data["audio_path"], target_sr=16000, channel_selector="average").samples
    # Match features dtype to model parameters (e.g., bf16) to avoid dtype mismatch in conv layers
    model_dtype = next(model.parameters()).dtype
    features = (
        model.processor([feature], sampling_rate=16000, return_tensors="pt").input_features.to(device=device, dtype=model_dtype)
    )

    # 7) Build transcription ids
    batch_transcription_ids = [tok.encode(transcriptions[0], add_special_tokens=False, return_tensors="pt").long().to(device)]

    return ctx, features, batch_transcription_ids, batch_start_positions, tok

def compute_z_desta(
    model: DeSTA25AudioModel,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: unkeHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        # Fallback: infer vocab size from lm_head weight shape
        lm_b = next(model.parameters()).new_zeros(lm_w.shape[1])

    # Build device
    device = next(model.parameters()).device
    print(f"[UNKE] compute_z_desta: computing right vector at layer {layer}")

    # Tokenize target (answer) into token IDs
    target_ids = tok(data["edited_answer"], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids.numel() > 0 and (target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id):
        target_ids = target_ids[1:]

    # Build DeSTA audio-aware context (tokens + start positions) and audio features
    ctx, features, batch_transcription_ids, batch_start_positions, _ = build_desta_audio_context_and_features(
        model=model, data=data, device=device
    )

    # Append answer tokens (teacher-forcing for next-token prediction)
    new_input_ids = torch.cat([ctx["input_ids"], target_ids[:-1].unsqueeze(0)], dim=1)
    new_attention_mask = torch.cat(
        [
            ctx["attention_mask"],
            torch.ones((ctx["attention_mask"].shape[0], target_ids.shape[0] - 1), device=device, dtype=ctx["attention_mask"].dtype),
        ],
        dim=1,
    )

    # Loss targets: ignore everywhere (-100) except the appended answer span
    rewriting_targets = torch.full(
        (new_input_ids.size(0), new_input_ids.size(1)), -100, device=device, dtype=torch.long
    )
    start = new_input_ids.size(1) - target_ids.size(0)
    rewriting_targets[:, start:] = target_ids

    # z edit location index for lookup
    lookup_idxs = [start]

    loss_layer = max(hparams.v_loss_layer, layer)

    # Delta dimensionality inferred from LLM hidden size
    if hasattr(model.config, "n_embd"):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=device)
    elif hasattr(model.config, "hidden_size"):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    elif hasattr(model.config.llm_config, "hidden_size"):
        delta = torch.zeros((model.config.llm_config.hidden_size,), requires_grad=True, device=device)
    else:
        raise NotImplementedError("Cannot determine hidden size for delta initialization")
    target_init = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        return cur_out

    # Optimizer on delta
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in tqdm(range(hparams.v_num_grad_steps), desc="compute_z"):
        opt.zero_grad()

        # Forward propagation with audio-aware inputs
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            _ = model(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                batch_features=features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
            )

        # Compute loss on rewriting targets
        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output

        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2
        )
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        total_loss = nll_loss + weight_decay.to(nll_loss.device)
        
        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate and step
        total_loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    
    # Cleanup optimization artifacts to prevent memory leaks
    del opt, delta
    
    # Ensure model parameters don't retain gradient state
    nethook.set_requires_grad(False, model)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
            
    torch.cuda.empty_cache()
    
    return target

def build_desta_batch_inputs(
    model: DeSTA25AudioModel, batch_data: List[Dict], device: torch.device
):
    """
    Build batched inputs for DeSTA following its preprocessing; returns
    (ctx, features, batch_transcription_ids, batch_start_positions)
    """
    # Ensure tokenizer/processor
    tok, processor = _ensure_desta_resources(model)

    audio_context_list = []
    start_positions_list = []
    transcriptions = []

    for d in batch_data:
        trans = d.get("transcription", "")
        transcriptions.append(trans)

        messages = [
            # {"role": "system", "content": "Focus on the audio clips and instructions."},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{d['question']}",
                "audios": [{"audio": d["audio_path"], "text": trans}],
            },
        ]
        audio_context = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audio_context = audio_context.replace(
            model.config.audio_locator, f"<start_audio>{model.config.audio_locator}<end_audio>"
        )
        token_list = tok.tokenize(audio_context)
        token_list, start_positions = _prepare_audio_context_and_start_positions(
            token_list=token_list,
            audio_locator=model.config.audio_locator,
            audio_size_list=[model.config.prompt_size],
            transcription_size_list=[len(tok.tokenize(trans, add_special_tokens=False))],
            placeholder_token=model.config.placeholder_token,
        )
        audio_context_list.append(tok.convert_tokens_to_string(token_list))
        start_positions_list.append(start_positions)

    ctx = tok(
        audio_context_list,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_length=True,
        add_special_tokens=False,
    ).to(device)

    batch_start_positions = []
    for i in range(ctx["length"].size(0)):
        total_length = ctx["length"][i]
        pad_length = (total_length - ctx["attention_mask"][i].sum()).item()
        for s in start_positions_list[i]:
            batch_start_positions.append((i, s + pad_length))

    # Audio features
    feats = []
    for d in batch_data:
        feat = AudioSegment.from_file(d["audio_path"], target_sr=16000, channel_selector="average").samples
        feats.append(feat)
    # Match features dtype to model parameters (e.g., bf16) to avoid dtype mismatch in conv layers
    model_dtype = next(model.parameters()).dtype
    features = (
        processor(feats, sampling_rate=16000, return_tensors="pt").input_features.to(device=device, dtype=model_dtype)
    )

    batch_transcription_ids = [
        tok.encode(t, add_special_tokens=False, return_tensors="pt").long().to(device)
        for t in transcriptions
    ]

    return ctx, features, batch_transcription_ids, batch_start_positions

def compute_ks_desta(
    model: DeSTA25AudioModel,
    tok: AutoTokenizer,
    batch_data: List[Dict],
    hparams: unkeHyperParams,
    layer: int,
):
    device = next(model.parameters()).device
    ctx, features, batch_transcription_ids, batch_start_positions = build_desta_batch_inputs(model, batch_data, device)
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = model(
                input_ids=ctx["input_ids"],
                attention_mask=ctx["attention_mask"],
                batch_features=features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
            )
            zs_out = tr.output
    zs_out = zs_out[0] if isinstance(zs_out, tuple) else zs_out
    idxs = [int(i.sum().item()) - 1 for i in ctx["attention_mask"]]
    zs_pick = torch.stack([zs_out[i, idxs[i]] for i in range(len(zs_out))], dim=0)
    return zs_pick, idxs

# ========================= Qwen2-Audio support =========================

def build_qwen_audio_inputs(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    data: Dict,
    device: torch.device,
):
    """
    Build Qwen2-Audio inputs (with audio) and a text context string for token ops.
    data expects: {"question": str, "audio_path": str, "edited_answer": str (optional)}
    """
    # Build conversation in Qwen2-Audio style (content list with audio+text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": data["audio_path"]},
                {"type": "text", "text": data["question"]},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    # Prepare audio features via processor by loading waveform arrays
    audio_arrays = []
    audio_path = data["audio_path"]

    import librosa  # type: ignore
    wav, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    if sr != processor.feature_extractor.sampling_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=processor.feature_extractor.sampling_rate)
    audio_arrays.append(wav)

    inputs = processor(
        text=text,
        audio=audio_arrays,
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )

    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    # Build token ids from tokenizer for appending target ids
    ctx_tok = tokenizer(text, return_tensors="pt")
    ctx_tok = {k: v.to(device) for k, v in ctx_tok.items()}
    return inputs, ctx_tok


def compute_z_qwen(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: unkeHyperParams,
):
    # Get model params for vocab projection (robust to path differences)
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    lm_w_param = None
    candidate_w_paths = [f"{hparams.lm_head_module}.weight", "model.lm_head.weight", "lm_head.weight"]
    for path in candidate_w_paths:
        try:
            lm_w_param = nethook.get_parameter(model, path)
            break
        except LookupError:
            continue
    if lm_w_param is None:
        # Fallback: scan parameters
        for n, p in model.named_parameters():
            if p.ndim == 2 and ("lm_head" in n or n.endswith("lm_head.weight")):
                lm_w_param = p
                break
    if lm_w_param is None:
        raise LookupError("Could not locate lm_head weight for Qwen2-Audio")
    lm_w = lm_w_param.T

    lm_b_param = None
    candidate_b_paths = [f"{hparams.lm_head_module}.bias", "model.lm_head.bias", "lm_head.bias"]
    for path in candidate_b_paths:
        try:
            lm_b_param = nethook.get_parameter(model, path)
            break
        except LookupError:
            continue
    if lm_b_param is None:
        lm_b = next(model.parameters()).new_zeros(lm_w.shape[1])
    else:
        lm_b = lm_b_param

    device = next(model.parameters()).device

    # Target token ids (answer)
    target_ids = tokenizer(data["edited_answer"], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids.numel() > 0 and (target_ids[0] == tokenizer.bos_token_id or target_ids[0] == tokenizer.unk_token_id):
        target_ids = target_ids[1:]

    inputs, ctx_tok = build_qwen_audio_inputs(model, processor, tokenizer, data, device)
    # Append answer tokens (teacher forcing) to the already-processed ids to keep audio-expanded
    # sequence length aligned with the model's forward pass.
    base_input_ids = inputs["input_ids"]
    base_attention_mask = inputs["attention_mask"]
    new_input_ids = torch.cat([base_input_ids, target_ids[:-1].unsqueeze(0)], dim=1)
    new_attention_mask = torch.cat(
        [
            base_attention_mask,
            torch.ones(
                (base_attention_mask.size(0), target_ids.size(0) - 1),
                dtype=base_attention_mask.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    # Loss targets mask
    rewriting_targets = torch.full(
        (new_input_ids.size(0), new_input_ids.size(1)), -100, device=device, dtype=torch.long
    )
    start = new_input_ids.size(1) - target_ids.size(0)
    rewriting_targets[:, start:] = target_ids

    lookup_idxs = [start]
    loss_layer = max(hparams.v_loss_layer, layer)

    # init delta (match the model hidden size, which is the OUT features of down_proj)
    down_proj_w = nethook.get_parameter(
        model, f"{hparams.layer_module_tmp.format(layer)}.mlp.down_proj.weight"
    )
    hidden_size = down_proj_w.shape[0]
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device)
    target_init = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in tqdm(range(hparams.v_num_grad_steps), desc="compute_z_qwen"):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            # Merge inputs with overridden ids/masks
            run_inputs = dict(inputs)
            run_inputs["input_ids"] = new_input_ids
            run_inputs["attention_mask"] = new_attention_mask
            _ = model(**run_inputs)

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        total_loss = nll_loss + weight_decay.to(nll_loss.device)

        if it == hparams.v_num_grad_steps - 1:
            break

        total_loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    
    # Cleanup optimization artifacts to prevent memory leaks
    del opt, delta
    
    # Ensure model parameters don't retain gradient state
    nethook.set_requires_grad(False, model)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
            
    torch.cuda.empty_cache()
    
    return target


def compute_ks_qwen_audio(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    batch_data: List[Dict],
    hparams: unkeHyperParams,
    layer: int,
):
    device = next(model.parameters()).device
    zs = []
    idxs = []
    for d in batch_data:
        inputs, ctx_tok = build_qwen_audio_inputs(model, processor, tokenizer, d, device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**inputs)
                out = tr.output
        out = out[0] if isinstance(out, tuple) else out
        # Use the attention mask from processor-expanded inputs to find the last token index
        last_idx = int(inputs["attention_mask"].sum().item()) - 1
        zs.append(out[0, last_idx])
        idxs.append(last_idx)
    return torch.stack(zs, dim=0), idxs


# ========================= Audio-Flamingo-3 support =========================

def build_af3_audio_inputs(
    model,  # AudioFlamingo3ForConditionalGeneration
    processor,  # AutoProcessor
    data: Dict,
    device: torch.device,
):
    """
    Build Audio-Flamingo-3 inputs with audio for UNKE.
    
    data expects: {"question": str, "audio_path": str, "edited_answer": str (optional)}
    Returns: processed inputs dict
    """
    from transformers import AudioFlamingo3ForConditionalGeneration
    
    # Build conversation in AF3 format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": data["question"]},
                {"type": "audio", "path": data["audio_path"]},
            ],
        }
    ]
    
    # Process using the processor's apply_chat_template
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    
    # Move tensors to device and handle dtype
    model_dtype = next(model.parameters()).dtype
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                processed_inputs[k] = v.to(device, dtype=model_dtype)
            else:
                processed_inputs[k] = v.to(device)
        else:
            processed_inputs[k] = v
    
    return processed_inputs


def compute_z_af3(
    model,  # AudioFlamingo3ForConditionalGeneration
    processor,  # AutoProcessor
    tokenizer,
    data: Dict,
    layer: int,
    hparams,  # AF3unkeHyperParams
):
    """
    Compute the target z vector for UNKE on Audio-Flamingo-3.
    """
    from transformers import AudioFlamingo3ForConditionalGeneration
    
    # Get model params for vocab projection
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    
    # Find lm_head weight
    lm_w_param = None
    candidate_w_paths = [f"{hparams.lm_head_module}.weight", "language_model.lm_head.weight"]
    for path in candidate_w_paths:
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

    # Find lm_head bias (may not exist)
    lm_b_param = None
    candidate_b_paths = [f"{hparams.lm_head_module}.bias", "language_model.lm_head.bias"]
    for path in candidate_b_paths:
        try:
            lm_b_param = nethook.get_parameter(model, path)
            break
        except LookupError:
            continue
    if lm_b_param is None:
        lm_b = next(model.parameters()).new_zeros(lm_w.shape[1])
    else:
        lm_b = lm_b_param

    device = next(model.parameters()).device

    # Target token ids (answer)
    target_ids = tokenizer(data["edited_answer"], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids.numel() > 0 and (target_ids[0] == tokenizer.bos_token_id or target_ids[0] == tokenizer.unk_token_id):
        target_ids = target_ids[1:]

    # Build audio inputs
    inputs = build_af3_audio_inputs(model, processor, data, device)
    
    # Append answer tokens (teacher forcing)
    base_input_ids = inputs["input_ids"]
    base_attention_mask = inputs["attention_mask"]
    new_input_ids = torch.cat([base_input_ids, target_ids[:-1].unsqueeze(0)], dim=1)
    new_attention_mask = torch.cat(
        [
            base_attention_mask,
            torch.ones(
                (base_attention_mask.size(0), target_ids.size(0) - 1),
                dtype=base_attention_mask.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    # Loss targets mask
    rewriting_targets = torch.full(
        (new_input_ids.size(0), new_input_ids.size(1)), -100, device=device, dtype=torch.long
    )
    start = new_input_ids.size(1) - target_ids.size(0)
    rewriting_targets[:, start:] = target_ids

    lookup_idxs = [start]
    loss_layer = max(hparams.v_loss_layer, layer)

    # Init delta
    down_proj_w = nethook.get_parameter(
        model, f"{hparams.layer_module_tmp.format(layer)}.mlp.down_proj.weight"
    )
    hidden_size = down_proj_w.shape[0]
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device)
    target_init = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in tqdm(range(hparams.v_num_grad_steps), desc="compute_z_af3"):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            run_inputs = dict(inputs)
            run_inputs["input_ids"] = new_input_ids
            run_inputs["attention_mask"] = new_attention_mask
            _ = model(**run_inputs)

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        total_loss = nll_loss + weight_decay.to(nll_loss.device)

        if it == hparams.v_num_grad_steps - 1:
            break

        total_loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    
    # Cleanup
    del opt, delta
    nethook.set_requires_grad(False, model)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
    torch.cuda.empty_cache()
    
    return target


def compute_ks_af3_audio(
    model,  # AudioFlamingo3ForConditionalGeneration
    processor,  # AutoProcessor
    tokenizer,
    batch_data: List[Dict],
    hparams,  # AF3unkeHyperParams
    layer: int,
):
    """Compute the current z vectors (before editing) for AF3 with audio inputs."""
    device = next(model.parameters()).device
    zs = []
    idxs = []
    for d in batch_data:
        inputs = build_af3_audio_inputs(model, processor, d, device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**inputs)
                out = tr.output
        out = out[0] if isinstance(out, tuple) else out
        last_idx = int(inputs["attention_mask"].sum().item()) - 1
        zs.append(out[0, last_idx])
        idxs.append(last_idx)
    return torch.stack(zs, dim=0), idxs