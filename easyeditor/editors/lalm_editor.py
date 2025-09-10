from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from ..dataset.processor.llavaov_processors import LLaVAOneVisionProcessor
from ..dataset.processor.qwen2vl_processors import Qwen2VLProcessor
from .editor import BaseEditor
import os.path
import os
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import json
import torch
import math
import random
import logging
import numpy as np
import librosa
import wandb
import jsonlines
import random

random.seed(42)

from ..trainer.utils import dict_to


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor

from ..util.globals import *
from .batch_editor import BatchEditor
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class LALMEditor:
    """LALM editor for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self, hparams: HyperParams):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_LALM_DICT[hparams.alg_name] if hparams.alg_name != "IKE" else None
        self.alg_name = hparams.alg_name
        self.hparams = hparams
        self.audio_root = hparams.audio_root 
        self.max_new_tokens = 256
        
        if hparams.alg_name == "IKE":
            if "qwen2-audio" in hparams.model_name.lower():
                self.training_set = Qwen2AudioDataset(
                    hparams.training_set_path, config=hparams
                )
            elif "desta" in hparams.model_name.lower():
                self.training_set = DeSTA25AudioDataset(
                    hparams.training_set_path, config=hparams
                )
        else:
            self.training_set = None

        make_logs()

        LOG.info("Instantiating model")
        
        if hparams.wandb_enabled:
            wandb.init(
                project=getattr(hparams, "wandb_project", "easyedit"),
                name=getattr(hparams, "wandb_run_name", None),
                config=hparams,
                reinit=True,
                group=""
            )
            self.wandb_enabled = True
            LOG.info(f"WandB enabled!")
        else:
            self.wandb_enabled = False

        if "qwen2-audio" in hparams.model_name.lower():
            from transformers import Qwen2AudioForConditionalGeneration
            
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=hparams.cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=hparams.cache_dir).tokenizer
        elif "desta" in hparams.model_name.lower():
            from desta import DeSTA25AudioModel
            
            self.model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", cache_dir=hparams.cache_dir, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct", cache_dir=hparams.cache_dir)
        else:
            raise NotImplementedError(f"Model {hparams.model_name} is not supported yet.")
        
    def _reload_model(self):
        del self.model
        torch.cuda.empty_cache()
        if "qwen2-audio" in self.model_name.lower():
            from transformers import Qwen2AudioForConditionalGeneration
            
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=self.hparams.cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
        elif "desta" in self.model_name.lower():
            from desta import DeSTA25AudioModel
            
            self.model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", cache_dir=self.hparams.cache_dir, device_map="auto")
        else:
            raise NotImplementedError(f"Model {self.model_name} is not supported yet.")
        
    def _get_icl_examples_from_training_set(self, edits: List[Tuple[str, str]]) -> List[dict]:
        """
        Used in IKE only.
        Randomly get a sample from the training set for each edit (pre_edit, post_edit).
        """
        examples = []
        for pre_edit, post_edit in edits:
            candidates = [
                item for item in self.training_set 
                if item['reliability']['original_answer'] == pre_edit 
                and item['reliability']['answer'] == post_edit
            ]
            if len(candidates) == 0:
                raise ValueError(f"No candidate found for edit ({pre_edit}, {post_edit}) in the training set.")
            example = random.choice(candidates)
            examples.append(example)
        return examples
            
    
    def single_edit_dataset(self, ds: Union[DeSTA25AudioDataset, Qwen2AudioDataset], output_path: str, generate_pre_edit: bool = True) -> None:
        LOG.info(f"Start editing the dataset and save results to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, mode='w') as writer:
            for i, sample in enumerate(tqdm(ds, desc='Test Editing samples', dynamic_ncols=True)):
                start_time = time()

                if generate_pre_edit:
                    LOG.info(f"Generating pre-editing results for sample {i}...")    
                    pre_edit = {}
                    for key in sample:
                        response = ds.generate_response(self.model, sample[key])
                        pre_edit[key] = response[0] if isinstance(response, list) else response
                    
                LOG.info(f"Performing editing for sample {i}...")
                if self.hparams.alg_name == "IKE":
                    edits = [(sample["reliability"]["original_answer"], sample["reliability"]["answer"])]
                    if self.hparams.use_icl_examples:
                        icl_examples = self._get_icl_examples_from_training_set(edits)
                        assert len(icl_examples) == 1, f"For now we only support one edit example for IKE."
                        icl_examples = list(icl_examples[0].values())
                    else:
                        icl_examples = []
                    
                    LOG.info(f"Generating post-editing results for sample {i}...")                
                    post_edit = {}
                    for key in sample:
                        response = ds.generate_ike_response(
                            self.model, 
                            icl_examples + [sample[key]],
                            edits=edits
                        )
                        post_edit[key] = response[0] if isinstance(response, list) else response
                else:
                    tokenized_reliability = ds.process_and_tokenize_batch([sample], key='reliability')
                    edited_model, weights_copy = self.apply_algo(
                        requests=tokenized_reliability,
                        hparams=self.hparams,
                        model=self.model,
                        copy=True,
                        return_orig_weights=True,
                    )
                    
                    LOG.info(f"Generating post-editing results for sample {i}...")                
                    post_edit = {}
                    for key in sample:
                        response = ds.generate_response(edited_model, sample[key])
                        post_edit[key] = response[0] if isinstance(response, list) else response
                        
                    LOG.info(f"Restoring original weights for sample {i}...")
                    # restore the original weights
                    for name, param in self.model.named_parameters():
                        if name in weights_copy:
                            param.data.copy_(weights_copy[name])
                            
                    del weights_copy
                    del edited_model
                        
                d = {
                    **sample,
                    "pre_edit": pre_edit if generate_pre_edit else {},
                    "post_edit": post_edit
                }
                
                LOG.info(f"Results for sample {i}: {d}")
                
                writer.write(d)
                
                if self.wandb_enabled:
                    wandb.log({
                        "step": i,
                        "edit_time": time() - start_time
                    })
    def sequential_edit_dataset(self, ds: Union[DeSTA25AudioDataset, Qwen2AudioDataset], output_path: str, generate_pre_edit: bool = True) -> None:
        """
        This will perform sequential editing for each sample in the dataset.
        The weights will not be restored after each edit until all edits for the sample are done.
        
        Before starting the sequential editing, the responses for the original model will be generated if generate_pre_edit is True.
        
        Before each edit i, we wil generate the pre-editing response for edit sample i.
        After each edit i, we will generate the post-editing response for edit sample from 0 to i.
        
        """
        LOG.info(f"Start editing the dataset and save results to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, mode='w') as writer:
            # Generate pre-editing results for all samples if needed
            if generate_pre_edit:
                LOG.info(f"Generating original pre-editing results for all samples...")
                pre_edit_all = []
                for i, sample in enumerate(tqdm(ds, desc='Generating pre-editing results', dynamic_ncols=True)):
                    pre_edit = {}
                    for key in sample:
                        response = ds.generate_response(self.model, sample[key])
                        pre_edit[key] = response[0] if isinstance(response, list) else response
                    pre_edit_all.append(pre_edit)
                writer.write({"pre_edit_all": pre_edit_all})
                LOG.info(f"Finished generating pre-editing results for all samples.")
            
            all_ike_edits = []
            all_ike_examples = []
            for i, sample in enumerate(tqdm(ds, desc='Test Editing samples', dynamic_ncols=True)):
                start_time = time()

                LOG.info(f"Generating intermediate pre-editing results for sample {i}...")    
                pre_edit = {}
                for key in sample:
                    if self.hparams.alg_name == "IKE":
                        response = ds.generate_ike_response(
                            self.model, 
                            all_ike_examples + [sample[key]],
                            edits=all_ike_edits
                        )
                    else:
                        response = ds.generate_response(self.model, sample[key])
                    pre_edit[key] = response[0] if isinstance(response, list) else response
                    
                LOG.info(f"Performing editing for sample {i}...")
                if self.hparams.alg_name == "IKE":
                    curr_edit = [(sample["reliability"]["original_answer"], sample["reliability"]["answer"])]
                    all_ike_edits.extend(curr_edit)
                    if self.hparams.use_icl_examples:
                        icl_examples = self._get_icl_examples_from_training_set(curr_edit)
                        assert len(icl_examples) == 1, f"For now we only support one edit example for IKE."
                        all_ike_examples.extend(list(icl_examples[0].values()))
                else:
                    tokenized_reliability = ds.process_and_tokenize_batch([sample], key='reliability')
                    edited_model, _ = self.apply_algo(
                        requests=tokenized_reliability,
                        hparams=self.hparams,
                        model=self.model,
                        copy=False,
                        return_orig_weights=False,
                        reset_model= i == 0
                    )

                    self.model = edited_model
                    
                post_edit = []
                for j, prev_sample in enumerate(tqdm(ds[:i+1], desc=f'Generating post-editing results for samples 0 to {i}', dynamic_ncols=True)):
                    post_edit_j = {}
                    for key in prev_sample:
                        if self.hparams.alg_name == "IKE":
                            response = ds.generate_ike_response(
                                self.model, 
                                all_ike_examples + [prev_sample[key]],
                                edits=all_ike_edits
                            )
                        else:
                            response = ds.generate_response(edited_model, prev_sample[key])
                        post_edit_j[key] = response[0] if isinstance(response, list) else response
                    post_edit.append(post_edit_j)
                        
                d = {
                    **sample,
                    "pre_edit": pre_edit,
                    "post_edit": post_edit
                }
                
                LOG.info(f"Results for sample {i}: {d}")
                
                writer.write(d)
                
                if self.wandb_enabled:
                    wandb.log({
                        "step": i,
                        "edit_time": time() - start_time
                    })

        # finally, restore the original model
        self._reload_model()