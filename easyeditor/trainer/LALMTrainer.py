from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from tqdm import tqdm

LOG = logging.getLogger(__name__)


class LALMTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_loc_output = {
                k: self.model(**batch[k]) for k in batch.keys() if k.startswith("locality")
            }
            
            base_loc_logits = {
                k: v.logits if not isinstance(v, torch.Tensor) else v
                for k, v in base_loc_output.items()
            }
            
            
            # base_outputs = self.model(batch["loc"])
            # if not isinstance(base_outputs, torch.Tensor):
            #     base_logits = base_outputs.logits
            # else:  
            #     base_logits = base_outputs
                
            # base_image_outputs = self.model(batch["loc_image"])
            # if not isinstance(base_image_outputs, torch.Tensor):
            #     base_image_logits = base_image_outputs.logits
            # else:
            #     base_image_logits = base_image_outputs
        
        # Do the edit

        start = time.time()
        # edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edited_model, model_info = self.model.edit(batch["reliability"])
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            ## Generality
            # post_edit_outputs = edited_model(batch["edit_outer"])
            # if not isinstance(post_edit_outputs, torch.Tensor):
            #     post_edit_logits = post_edit_outputs.logits
            #     post_batch_labels = post_edit_outputs.labels
            # else:
            #     post_edit_logits = post_edit_outputs
            #     post_batch_labels = batch["edit_outer"]["labels"]

            post_edit_generality_outputs = {
                k: edited_model(**batch[k]) for k in batch.keys() if k.startswith("generality")
            }
            
            post_edit_generality_logits_labels = {
                k: {
                    "logits": v.logits if not isinstance(v, torch.Tensor) else v,
                    "labels": v.labels if not isinstance(v, torch.Tensor) else batch[k]["labels"]
                } for k, v in post_edit_generality_outputs.items()
            }
            
            # Reliability after edit
            inner_edit_outputs = edited_model(**batch["reliability"])
            post_edit_reliability_logits_labels = {
                "logits": inner_edit_outputs.logits if not isinstance(inner_edit_outputs, torch.Tensor) else inner_edit_outputs,
                "labels": inner_edit_outputs.labels if not isinstance(inner_edit_outputs, torch.Tensor) else batch["reliability"]["labels"]
            }
            # inner_edit_outputs = edited_model(batch["edit_inner"])
            
            # if not isinstance(inner_edit_outputs, torch.Tensor):
            #     inner_edit_logits = inner_edit_outputs.logits
            #     inner_batch_labels = inner_edit_outputs.labels
            # else:
            #     inner_edit_logits = inner_edit_outputs
            #     inner_batch_labels = batch["edit_inner"]["labels"]

            # rephrase image
            # if self.train_set.__class__.__name__ == "ComprehendEditDataset":
            #     post_image_edit_logits = inner_edit_logits
            #     post_image_batch_labels = inner_batch_labels
            # else:
            #     post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            #     if not isinstance(post_image_edit_outputs, torch.Tensor):
            #         post_image_edit_logits = post_image_edit_outputs.logits
            #         post_image_batch_labels = post_image_edit_outputs.labels
            #     else:
            #         post_image_edit_logits = post_image_edit_outputs
            #         post_image_batch_labels = batch["edit_outer_image"]["labels"]

            # Generality loss
            
            generality_loss = {
                k: self.model.edit_loss_fn(self.config, v["logits"], v["labels"])['nll'] for k, v in post_edit_generality_logits_labels.items()
            }
            
            # l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)["nll"]
            # l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)["nll"]          
            
            # Collect some useful metrics
            with torch.no_grad():
                post_edit_dict = {
                    k: self.model.edit_loss_fn(self.config, v["logits"], v["labels"]) for k, v in post_edit_generality_logits_labels.items()
                }
                inner_edit_dict = self.model.edit_loss_fn(self.config, post_edit_reliability_logits_labels["logits"], post_edit_reliability_logits_labels["labels"])
                

            # post_base_outputs = edited_model(batch["loc"])
            # if not isinstance(post_base_outputs, torch.Tensor):
            #     post_base_logits = post_base_outputs.logits
            #     kl_mask = post_base_outputs.attention_mask
            # else:
            #     post_base_logits = post_base_outputs
            #     kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            # post_image_base_outputs = edited_model(batch["loc_image"])
            # if not isinstance(post_base_outputs, torch.Tensor):
            #     post_image_base_logits = post_image_base_outputs.logits
            #     kl_image_mask = post_image_base_outputs.attention_mask
            # else:
            #     post_image_base_logits = post_image_base_outputs
            #     kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)
            
            ### Locality KL loss
            post_locality_outputs = {
                k: edited_model(**batch[k], return_logits_only=False) for k in batch.keys() if k.startswith("locality")
            }
            post_locality_logits = {
                k: v.logits if not isinstance(v, torch.Tensor) else v
                for k, v in post_locality_outputs.items()
            }
            
            # Collect attention mask for kl loss computation
            if 'qwen2-audio' in self.config.model_name.lower():
                kl_masks = {
                    k: post_locality_outputs[k].attention_mask for k in post_locality_outputs.keys()
                }
            elif 'desta' in self.config.model_name.lower():
                kl_masks = {
                    k: batch[k]["attention_mask"] for k in batch.keys() if k.startswith("locality") # Should be prepared already in the batch
                }
            else:
                LOG.info("No attention mask found for locality KL loss, using ones")
                kl_masks = {
                    k: torch.ones(v.shape[0], v.shape[1]).to(v.device) for k, v in post_locality_logits.items()
                }

            # Compute KL loss
            kl_losses = {
                k: kl_loc_loss(base_loc_logits[k].detach(), post_locality_logits[k], mask=kl_masks[k]) for k in base_loc_logits.keys()
            }
            # l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            # l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        # if l_edit.isnan():
        #     print("l_edit is nan")
        #     print("input: ", batch["edit_outer"]['text_input'])
        # elif l_image_edit.isnan():
        #     print("l_image_edit is nan")
        #     print("input: ", batch["edit_outer_image"]['text_input'])
        # elif l_loc.isnan():
        #     print("l_loc is nan")
        #     print("input: ", batch["loc"]['text_input'])
        # elif l_image_loc.isnan():
        #     print("l_image_loc is nan")
        #     print("input: ", batch["loc_image"]['text_input'])

        # if self.config.alg == "SERAC_MULTI":
        #     l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        # else:
        #     l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * l_image_edit
        
        l_edit = sum(v for v in generality_loss.values())
        l_loc = sum(v for v in kl_losses.values())
        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # # Text locality
        # post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        # base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        # # Image locality
        # post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
        # base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
        
        post_locality_logits_softmax_top_k = {
            k: torch.topk(torch.nn.functional.softmax(post_locality_logits[k], dim=-1), k=1, dim=-1).indices
            for k in post_locality_logits.keys()
        }

        base_locality_logits_softmax_top_k = {
            k: torch.topk(torch.nn.functional.softmax(base_loc_logits[k], dim=-1), k=1, dim=-1).indices
            for k in base_loc_logits.keys()
        }
        
        
        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        # info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        for k in post_edit_dict:
            info_dict[f'edit/{k}_acc'] = post_edit_dict[k]["acc"].item()
            info_dict[f'edit/{k}_log_prob'] = post_edit_dict[k]["log_prob"].item()
            info_dict[f'edit/{k}_prob'] = post_edit_dict[k]["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        # info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        for k in post_locality_logits_softmax_top_k.keys():
            info_dict[f'loc/{k}_acc'] = sum(post_locality_logits_softmax_top_k[k].view(-1) == base_locality_logits_softmax_top_k[k].view(-1))/post_locality_logits_softmax_top_k[k].view(-1).shape[0]
        # info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        # info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        # outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        gen0_acc = f"{stats['edit/generality_type_0_acc_val']:<12.5f}"
        gen1_acc = f"{stats['edit/generality_type_1_acc_val']:<12.5f}"
        gen2_acc = f"{stats['edit/generality_type_2_acc_val']:<12.5f}"
        gen_accs = [gen0_acc, gen1_acc, gen2_acc]
        
        # image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        # loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        # loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"
        loc_audio0_acc = f"{stats['loc/locality_audio_type_0_acc_val']:<12.5f}"
        loc_audio1_acc = f"{stats['loc/locality_audio_type_1_acc_val']:<12.5f}"
        loc_audio2_acc = f"{stats['loc/locality_audio_type_2_acc_val']:<12.5f}"
        loc_audio_accs = [loc_audio0_acc, loc_audio1_acc, loc_audio2_acc]
        if 'loc/locality_audio_type_3_acc_val' in stats:
            loc_audio3_acc = f"{stats['loc/locality_audio_type_3_acc_val']:<12.5f}"
            loc_audio_accs.append(loc_audio3_acc)
        loc_text_acc = f"{stats['loc/locality_text_acc_val']:<12.5f}"

        LOG.info(
          f"Step {prog} generality_acc: {[gen0_acc, gen1_acc, gen2_acc]} inner_acc: {inner_acc} it_time: {elapsed:.4f} locality_audio_acc: {loc_audio_accs}, locality_text_acc: {loc_text_acc} "
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=self.config.silent, dynamic_ncols=True, )):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats