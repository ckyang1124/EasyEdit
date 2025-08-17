from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoProcessor
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc,
    test_batch_prediction_acc,
    test_prediction_acc,
    test_generation_quality,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)
from copy import deepcopy


# TODO: modify icl for lalm

# def compute_icl_lalm_edit_quality(
#         model,
#         model_name,
#         hparams: HyperParams,
#         tok: AutoTokenizer,
#         # vis_tok,
#         icl_examples,
#         record: typing.Dict,
#         device,
#         pre_edit: bool = False
# ) -> typing.Dict:
#     """
#     Given a rewritten model, computes generalization and specificity metrics for
#     the desired rewrite (passed in via the CounterFact dataset record). Returns a
#     dictionary containing those metrics.

#     :param model: Rewritten model
#     :param tok: Tokenizer
#     :param record: CounterFact dataset record
#     :param snips: ???
#     :param vec: ???
#     :return: Dictionary containing rewriting metrics
#     """
#     vis_root = hparams.coco_image
#     rephrase_root = hparams.rephrase_image
#     # First, unpack rewrite evaluation record.
#     target = record["target"]
#     prompt = record["prompt"]
#     image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
#     rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
#     rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
#     if rephrase_image is not None:
#         rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)

#     if "locality_prompt" in record.keys():
#         loc_q = record["locality_prompt"]
#         loc_a = record["locality_ground_truth"]
#     if "multimodal_locality_image" in record.keys():
#         m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(hparams.device)
#         m_loc_q = record["multimodal_locality_prompt"]
#         m_loc_a = record["multimodal_locality_ground_truth"]

#     new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

#     if pre_edit:
#         edit_acc, _ = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                              target, prompt, image)
#     else:
#         edit_acc, _ = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                              target, new_fact, image)
#     ret = {
#         f"rewrite_acc": edit_acc
#     }
#     if rephrase is not None:
#         rephrase_acc, _ = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                                  target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
#         ret['rephrase_acc'] = rephrase_acc

#     if "image_rephrase" in record.keys():
#         rephrase_image_acc, _ = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                                        target, new_fact, rephrase_image)
#         ret['rephrase_image_acc'] = rephrase_image_acc

#     if "locality_prompt" in record.keys():
#         if pre_edit:
#             _, _, locality_output = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                                            loc_a, loc_q, None, is_loc=True)
#         else:
#             _, _, locality_output = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                                            loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None, is_loc=True)
#         ret['locality_output'] = locality_output

#     if "multimodal_locality_image" in record.keys():
#         if pre_edit:
#             _, _, locality_image_output = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                                                  m_loc_a, m_loc_q, m_loc_image, is_loc=True)
#         else:
#             _, _, locality_image_output = icl_lalm_lm_eval(model, model_name, hparams, tok, icl_examples,
#                                                                  m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image, is_loc=True)
#         ret['multimodal_locality_output'] = locality_image_output

#     return ret

# def icl_lalm_lm_eval(
#         model,
#         model_name,
#         hparams: HyperParams,
#         tokenizer,
#         icl_examples,
#         target,
#         x,
#         image,
#         is_loc=False,
#         neighborhood=False )-> typing.Dict:
#     device = torch.device(f'cuda:{hparams.device}')

#     samples = prepare_lalm_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image)

#     # return compute_lalm_edit_quality(model, samples, hparams.exact_match)
#     return compute_lalm_edit_quality(model, samples,
#                                            hparams.exact_match) if not is_loc else compute_lalm_edit_quality_demo(
#         model, samples)

# TODO: remove it
decode_fn: Optional[typing.Callable] = None


# TODO: modify this function
# Will it output a word rather than a single token?
def compute_lalm_hf_edit_quality(model, orig_batch, exach_match=False):
    batch = deepcopy(orig_batch)
    input_length = batch["input_ids"].shape[1]
    print(f"---")
    print(f"Batch input_ids: {decode_fn(batch['input_ids'])[0] if decode_fn else batch['input_ids']}")
    print(f"Batch labels: {decode_fn(batch['labels'])[0] if decode_fn else batch['labels']}")
    print(f"batch.keys(): {batch.keys()}")
    
    # TODO: change mex_new_length, and detect eos token
    with torch.no_grad():
        for _ in range(16): # max_new_length: 32
            batch_without_labels = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model(**batch_without_labels)
            
            # Append the new token to the input_ids to keep generating
            batch["input_ids"] = torch.cat((batch["input_ids"], outputs.logits[:, -1, :].argmax(-1, keepdim=True)), dim=1)
            batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones((batch["attention_mask"].shape[0], 1), device=batch["attention_mask"].device)), dim=1)
            
            # if decode_fn:
                # print(decode_fn(outputs.logits[:, -1, :].argmax(-1, keepdim=True))[0], end = ' ')
        # print("\nGeneration complete.")
            
            
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = batch["labels"].cpu()
    
    # print(f"input_ids shape: {batch['input_ids'].shape}")
    # print(f"Logits shape: {logits.shape}, Target shape: {targ.shape}")
    # print(f"logits: {logits}")
    # print(f"targ: {targ}")
    
    # all_pred_ids = logits.argmax(-1)
    
    logits = logits[:, input_length - 1:, :]  # Slicing to remove the first input tokens and the last token logits 
    # Slicing to get the last token logits
    pred_ids = logits.argmax(-1)
    print(f"Predicted IDs: {decode_fn(pred_ids)[0] if decode_fn else pred_ids}")
    # print("After slicing logits and targ:")
    # print(f"Logits shape: {logits.shape}, Target shape: {targ.shape}")
    # print(f"Predicted IDs: {pred_ids}")
    # print(f"All Predicted IDs: {all_pred_ids}")
    # print(f"Decoded Input IDs: {decode_fn(batch['input_ids']) if decode_fn else batch['input_ids']}")
    # print(f"Decoded Predicted IDs: {decode_fn(all_pred_ids) if decode_fn else pred_ids}")
    return 0, pred_ids.numpy()
    
    if logits.dim() == 3:
        logits = logits[:, :-1, :]
        targ = targ[:, 1:]
        
    mask = targ != -100
    targ[~mask] = 0    
    if exach_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    pred_ids = pred_ids.masked_select(pred_ids != 0).view(1, -1)
    return acc, pred_ids.numpy()


def compute_lalm_edit_quality(model, batch, exact_match=False):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = outputs.labels.detach().cpu()

    if logits.dim() == 3:
        logits = logits[:, :-1]
        targ = targ[:, 1:]
        # logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    if exact_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    return acc, pred_ids.numpy()


def compute_lalm_hf_edit_results(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoProcessor,
        record: typing.Dict,
        device,
        curr_decode_fn: Optional[typing.Callable] = None
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    
    global decode_fn
    decode_fn = curr_decode_fn
    
    
    print("=" * 10 + " Reliability " + "=" * 10)
    ret['rewrite_acc'], ret['rewrite_output'] = compute_lalm_hf_edit_quality(model, record["reliability"])
    
    print("=" * 10 + " Locality " + "=" * 10)

    locality_info = [
        compute_lalm_hf_edit_quality(model, record[key])
        for key in record.keys()
        if key.startswith("locality")
    ]
    for i, info in enumerate(locality_info):
        ret[f'locality_type_{i}_output'] = info[1]
    
    print("=" * 10 + " Generality " + "=" * 10)
        
    generality_info = [
        compute_lalm_hf_edit_quality(model, record[key])
        for key in record.keys()
        if key.startswith("generality")
    ]
    for i, info in enumerate(generality_info):
        ret[f'generality_type_{i}_acc'] = info[0]
    for i, info in enumerate(generality_info):
        ret[f'generality_type_{i}_output'] = info[1]

    # TODO: check portability acc
    ret['portability_acc'], _ = compute_lalm_hf_edit_quality(model, record["portability_audio"])

    return ret