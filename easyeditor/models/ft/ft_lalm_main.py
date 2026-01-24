from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...trainer.utils import dict_to
from ...util import nethook
from ...trainer.utils import _logits
from ...trainer.losses import multiclass_log_probs, masked_log_probs

from .ft_lalm_hparams import FTLALMHyperParams

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def apply_ft_to_lalm(
    model,
    requests: dict,
    hparams: FTLALMHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    LOG.info(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    model,
    request: dict,
    hparams: FTLALMHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    
    request:
    {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        ...
        'labels': labels,
    }
    prompt and target should already be concated
    """
    request = dict_to(request, hparams.device)
    
    # Retrieve weights that user desires to change
    # weights = {
    #     n: p
    #     for n, p in model.named_parameters()
    #     for layer in hparams.layers
    #     if hparams.rewrite_module_tmp.format(layer) in n
    # }
    weights = {
        n: p
        for n, p in model.named_parameters()
        if any(
            n.startswith(inner_n)
            for inner_n in hparams.inner_params
        ) 
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    LOG.info(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        LOG.info(20 * "=")
        LOG.info(f"Epoch: {it}")
        LOG.info(20 * "=")
        loss_meter.reset()

        opt.zero_grad()
        if "desta" in hparams.model_name.lower():
            outputs = _logits(
                model(
                    input_ids=request['input_ids'], 
                    attention_mask=request['attention_mask'], 
                    batch_features=request['batch_features'], 
                    batch_transcription_ids=request['batch_transcription_ids'], 
                    batch_start_positions=request['batch_start_positions']
                )
            )
            loss = masked_log_probs(hparams, outputs, request['labels'], shift=True)["nll"]
        elif "qwen2-audio" in hparams.model_name.lower():
            outputs = _logits(
                model(
                    input_ids=request['input_ids'],  
                    input_features=request['input_features'], 
                    attention_mask=request['attention_mask'], 
                    feature_attention_mask=request['feature_attention_mask']
                )
            )
            loss = masked_log_probs(hparams, outputs, request['labels'], shift=True)["nll"]
        elif "audio-flamingo" in hparams.model_name.lower():
            input_features = request['input_features']
            if input_features is not None:
                orig_dtype = input_features.dtype
            if hasattr(model, 'dtype') and input_features is not None:
                input_features = input_features.to(model.dtype)
                
            outputs = _logits(
                model(
                    input_ids=request['input_ids'],
                    input_features=input_features,
                    attention_mask=request['attention_mask'], 
                    input_features_mask=request['input_features_mask']
                )
            )
            
            if input_features is not None:
                input_features = input_features.to(orig_dtype)
                
            loss = masked_log_probs(hparams, outputs, request['labels'], shift=True)["nll"]
        else:
            raise NotImplementedError
 
        LOG.info(f"Batch loss {loss.item()}")
        loss_meter.update(loss.item(), n=1)

        if loss.item() >= 1e-2:
            loss.backward()
            opt.step()

        if type(hparams.norm_constraint) is float:
            eps = hparams.norm_constraint
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = torch.clamp(
                        v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                    )

        LOG.info(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    LOG.info(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
