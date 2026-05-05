import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ...util import nethook
from .compute_z import compute_z_desta, compute_ks_desta
from .unke_hparams import unkeHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeHyperParams,
    layer: int,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to("cuda")
    idxs = [i.sum()-1 for i in input_ids['attention_mask']]
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list=[]
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i,idxs[i]])
    zs_out =torch.stack(zs_out_list,dim=0)


    return zs_out,idxs

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # and 'mlp' in n
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters




def apply_unke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: unkeHyperParams,
    batch_data: list,
    ex_data: list,
    audio_ex_data: list | None = None,
):

    # Detect DeSTA model instance or by arg_note
    inner_model = model.llm_model

    preserve_params = []
    for name, params in model.named_parameters():
        # Target only parameters belonging to selected decoder layers
        for layer_idx in hparams.layers:
            if f".layers.{layer_idx}." in name:
                preserve_params.append(name)
                break
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}




    z_layer = hparams.layers[-1]
    print(f"[UNKE] Selected edit layer (z_layer) = {z_layer}")
    z_list = []
    for data in batch_data:

        print(f"[UNKE] compute_z_desta at layer {z_layer}")
        cur_z = compute_z_desta(
            model,
            tok,
            data,
            z_layer,
            hparams,
        )

        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=0)#(bs,h_dim)
    #print(zs.shape)
    batch_question = [i['question'] for i in batch_data]
    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"[UNKE] Updating weights at layer {layer}")
        #print(f"\n\nLAYER {layer}\n")
        # Build audio-aware batch and trace
        from .compute_z import build_desta_batch_inputs
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
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        

        cur_zs, idxs = compute_ks_desta(model, tok, batch_data, hparams, z_layer)
        
        
        targets = zs - cur_zs 
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            # Trace inner LLM directly for text-only ex_data
            inner_layer = hparams.layer_module_tmp.format(layer).replace("llm_model.", "")
            with nethook.Trace(
                module=inner_model,
                layer=inner_layer,
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = inner_model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
            # If audio preservation examples are provided, trace full DeSTA with audio
            if audio_ex_data is not None and len(audio_ex_data) > 0:
                from .compute_z import build_desta_batch_inputs
                device = next(model.parameters()).device
                audio_ctx, audio_features, audio_batch_transcription_ids, audio_batch_start_positions = build_desta_batch_inputs(
                    model, audio_ex_data, device
                )
                with nethook.Trace(
                    module=model,
                    layer=hparams.layer_module_tmp.format(layer),
                    retain_input=True,
                    retain_output=True,
                    detach=True,
                    clone=True,
                ) as tr_audio:
                    _ = model(
                        input_ids=audio_ctx["input_ids"],
                        attention_mask=audio_ctx["attention_mask"],
                        batch_features=audio_features,
                        batch_transcription_ids=audio_batch_transcription_ids,
                        batch_start_positions=audio_batch_start_positions,
                    )
                    audio_stat_in = tr_audio.input
                    audio_stat_out = tr_audio.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out
        if 'audio_stat_out' in locals():
            audio_stat_out = audio_stat_out[0] if type(audio_stat_out) is tuple else audio_stat_out



        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers(1,4096)

        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,hparams.lr)
        
        
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        
        for i in range(len(idxs)):
            
            layer_out_ks[i,idxs[i]]+=resid[i]
        
        # get_qwen2_causal_mask
        # llama2
        if 'llama' in hparams.model_name.lower():
            # For DeSTA, use ctx attention mask; otherwise contexts_tok
            attn_mask_inputs = ctx['attention_mask']
            input_causal_mask, input_position_ids, input_cache_position = get_causal_mask(layer_in_ks, attn_mask_inputs)
            ex_causal_mask, ex_position_ids, ex_cache_position = get_causal_mask(stat_in,ex_tok['attention_mask'])
        elif 'qwen' in hparams.model_name.lower():
            attn_mask_inputs = ctx['attention_mask']
            input_causal_mask, input_position_ids = get_qwen2_causal_mask(layer_in_ks, attn_mask_inputs)
            ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(stat_in,ex_tok['attention_mask'])
        
        
        for step in tqdm(range(hparams.optim_num_step)):
            #scheduler.step()
            optimizer.zero_grad()
            if 'qwen' in hparams.model_name.lower():
                 # modeling_qwen2.py, line 551
                ex_inputs_embeds = inner_model.model.embed_tokens(ex_position_ids)
                ex_hidden_states = ex_inputs_embeds

                ex_position_embeddings = inner_model.model.rotary_emb(ex_hidden_states, ex_position_ids)

                loss1 = criterion(
                    _layer(
                        stat_in,
                        attention_mask=ex_causal_mask,
                        # position_ids=ex_position_ids,
                        position_embeddings=ex_position_embeddings
                    )[0],
                    stat_out
                )

                inputs_embeds = inner_model.model.embed_tokens(input_position_ids)
                hidden_states = inputs_embeds
                
                position_embeddings = inner_model.model.rotary_emb(hidden_states, input_position_ids)

                loss2 = criterion(
                    _layer(
                        layer_in_ks,
                        attention_mask=input_causal_mask,
                        # position_ids=input_position_ids,
                        position_embeddings=position_embeddings
                    )[0],
                    layer_out_ks
                )

                loss = loss1 + loss2
            else:  # Llama-style (including DeSTA's Llama backbone)
                ex_inputs_embeds = inner_model.model.embed_tokens(ex_position_ids)
                ex_hidden_states = ex_inputs_embeds

                ex_position_embeddings = inner_model.model.rotary_emb(ex_hidden_states, ex_position_ids)

                loss1 = criterion(
                    _layer(
                        stat_in,
                        attention_mask=ex_causal_mask,
                        # position_ids=ex_position_ids,
                        position_embeddings=ex_position_embeddings,
                        cache_position=ex_cache_position
                    )[0],
                    stat_out
                )

                inputs_embeds = inner_model.model.embed_tokens(input_position_ids)
                hidden_states = inputs_embeds
                
                position_embeddings = inner_model.model.rotary_emb(hidden_states, input_position_ids)

                loss2 = criterion(
                    _layer(
                        layer_in_ks,
                        attention_mask=input_causal_mask,
                        # position_ids=input_position_ids,
                        position_embeddings=position_embeddings,
                        cache_position=input_cache_position
                    )[0],
                    layer_out_ks
                )

                # Optional third loss term to preserve audio behavior using audio_ex_data
                if (audio_ex_data is not None and len(audio_ex_data) > 0):
                    audio_causal_mask, audio_position_ids, audio_cache_position = get_causal_mask(
                        audio_stat_in, audio_ctx['attention_mask']
                    )
                    audio_inputs_embeds = inner_model.model.embed_tokens(audio_position_ids)
                    audio_hidden_states = audio_inputs_embeds
                    audio_position_embeddings = inner_model.model.rotary_emb(audio_hidden_states, audio_position_ids)
                    loss_audio = criterion(
                        _layer(
                            audio_stat_in,
                            attention_mask=audio_causal_mask,
                            position_embeddings=audio_position_embeddings,
                            cache_position=audio_cache_position
                        )[0],
                        audio_stat_out
                    )
                    loss = loss1 + loss2 + loss_audio
                    # print("loss1", loss1.item())
                    # print("loss2", loss2.item())
                    # print("loss_audio", loss_audio.item())
                    # print("loss", loss.item())
                else:
                    loss = loss1 + loss2
           
            loss.backward(retain_graph=True)
            optimizer.step()    
            
            if getattr(hparams, "verbose", False):
                tqdm.write(f'Step [{step+1}/{hparams.optim_num_step}], Loss: {loss.item()}, Layer:{layer}')

            # if loss.item() < 5e-5:
            #     break

        # Reset gradients and requires_grad state for all layer parameters
        for n, m in _layer.named_parameters():
            if m.grad is not None:
                m.grad.data.zero_()
            m.requires_grad = False
        
        cleanup_tensors = [layer_in_ks, layer_out_ks, cur_zs, targets, stat_in, stat_out]
        if (audio_ex_data is not None and len(audio_ex_data) > 0):
            cleanup_tensors.extend([audio_stat_in, audio_stat_out])
        for x in cleanup_tensors:
            x.cpu()
            del x
        
        # Clear optimizer state to prevent memory accumulation
        del optimizer
        torch.cuda.empty_cache()
        
    return weights_copy

    
def get_qwen2_causal_mask(input_tensor,attention_mask,past_key_values_length = 0):
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_tensor.shape[0], input_tensor.shape[1]),
            input_tensor,
            0,
        )

    return attention_mask,position_ids

def get_causal_mask(input_tensor,attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position