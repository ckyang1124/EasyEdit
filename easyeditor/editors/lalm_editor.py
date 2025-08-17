from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from ..dataset.processor.llavaov_processors import LLaVAOneVisionProcessor
from ..dataset.processor.qwen2vl_processors import Qwen2VLProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch
import math
import random
import logging
import numpy as np
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor

from ..util.globals import *
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_hf_edit_results,
                        compute_lalm_hf_edit_results)
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
        self.apply_algo = ALG_LALM_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        self.hparams = hparams

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            
            if "qwen2-audio" in hparams.model_name.lower():
                from transformers import Qwen2AudioForConditionalGeneration
                
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=hparams.cache_dir, device_map="auto")
                self.tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=hparams.cache_dir).tokenizer
            else:
                from desta import DeSTA25AudioModel
                
                self.model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", cache_dir=hparams.cache_dir, device_map="auto")
                self.tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct", cache_dir=hparams.cache_dir)
        else:
            self.model, self.tokenizer = self.model_name
            
        # self.model.to(f'cuda:{hparams.device}')
        

    def edit(self,
            # prompts: Union[str, List[str]],
            # targets: Union[str, List[str]],
            # image: Union[str, List[str]],
            # file_type: Union[str,List[str]] = None,
            # rephrase_prompts: Optional[Union[str, List[str]]] = None,
            # rephrase_image: Optional[Union[str, List[str]]] = None,
            # locality_inputs: Optional[dict] = None,
            # edit_sample: Union[dict, List[dict]],
            testing_ds: Dataset,
            edit_sample: Union[dict, List[dict]] = None,
            keep_original_weight=False,
            verbose=True,
            sequential_edit=False,
            **kwargs
            ):
        """
        edit_sample: dict or list of dict {
            "audio": 
            "transcription":
            "reliability_question":
            "original_answer":
            "edited_answer":
            "generality": []
            "locality": {
                "audio": [],
                "text": [],
            }
            "portability": {
                "audio": {}
            }
        }
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            # TODO: I am not sure what is it for
            self.hparams.batch_size = 1
        
        assert testing_ds or edit_sample, \
            'You must provide either testing_ds or edit_sample to edit.'
        if testing_ds:
            assert edit_sample is None, 'You can not set both testing_ds and editing_sample at the same time.'
            test_loader = DataLoader(testing_ds, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=testing_ds.collate_fn)
        if edit_sample:
            # TODO
            raise NotImplementedError("Providing edit_sample directly is not supported yet. Please use testing_ds instead.")
            # assert testing_ds is None, 'You can not set both testing_ds and editing_sample at the same time.'
        if sequential_edit:
            # TODO
            raise NotImplementedError("Sequential editing is not supported yet. Please set sequential_edit=False.")
            
        # sequential editing
        if sequential_edit:
            # TODO: figure out what to do when sequential_edit is True
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                if self.alg_name == 'IKE' and self.hparams.k == 0:
                    edited_model = self.model
                    weights_copy = None
                else:
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tokenizer,
                        request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=None
                    )
            exec_time = time() - start
            if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
                print("Start saving the WISE model!")
                edited_model.save(self.hparams.save_path)
                
            all_metrics = []
            exec_time = time() - start
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                if self.alg_name == 'IKE':
                    if self.hparams.k != 0:    
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, icl_examples,
                                                                request, self.hparams.device),
                            "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, [''],
                                                                request, self.hparams.device, pre_edit=True)
                        }
                    else:
                        # QUESTION = request["prompt"]
                        # ANSWER = request["target"]
                        from copy import deepcopy
                        prompt_new_request = deepcopy(request)
                        prefix_template = self.hparams.template.format(prompt=request["prompt"],target=request["target"])
                        prompt_new_request["prompt"] = prefix_template + request["prompt"]
                        prompt_new_request["rephrase_prompt"] = prefix_template + request["rephrase_prompt"]
                        prompt_new_request["locality_prompt"] = prefix_template + request["locality_prompt"]
                        prompt_new_request["multimodal_locality_prompt"] = prefix_template + request["multimodal_locality_prompt"]
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
                                                                prompt_new_request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
                                                                request, self.hparams.device)
                        }
                else:
                    if self.model_name in ['minigpt4', 'blip2']:
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer,
                                                                request, self.hparams.device),
                            "pre": compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
                                                                request, self.hparams.device)
                        }
                    elif self.model_name in ['llava-onevision', 'qwen2-vl']:
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer,
                                                                request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
                                                                request, self.hparams.device)
                        }
                                
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = torch.tensor(metrics['pre']['locality_output']).to(torch.float32)
                    post_logits = torch.tensor(metrics['post']['locality_output']).to(torch.float32)
                    metrics['post']['locality_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = torch.tensor(metrics['pre']['multimodal_locality_output']).to(torch.float32)
                    post_image_logits = torch.tensor(metrics['post']['multimodal_locality_output']).to(torch.float32)
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_logits.view(-1) == base_image_logits.view(-1))/post_image_logits.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')

                LOG.info(f"Evaluation took {time() - start}")
                all_metrics.append(metrics)
        
        # single editing
        else:
            all_metrics = []
            for i, request in enumerate(tqdm(test_loader, disable=not verbose, desc='Editing dataset', dynamic_ncols=True)):
                if i > 1:
                    break
                
                tqdm.write("Collecting pre-editing results...")
                pre_edit_result = compute_lalm_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)    
                
                start = time()
                if self.alg_name == 'IKE':
                    # TODO: modify IKE for LALM in the future
                    # What is self.hparams.k?
                    # Is self.hparams.k the number of in-context examples?
                    if self.hparams.k != 0:                    
                        assert 'train_ds' in kwargs.keys(), 'IKE need train_ds (For getting In-Context prompt)'
                        edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                            self.model,
                            self.tokenizer,
                            request,
                            self.hparams,
                            copy=False,
                            return_orig_weights=True,
                            keep_original_weight=keep_original_weight,
                            train_ds=kwargs['train_ds']
                        )
                    else:
                        edited_model = self.model
                        weights_copy = None
                else:
                    edited_model, weights_copy = self.apply_algo(
                        # self.tokenizer,
                        requests=request["reliability"], # [request],
                        hparams=self.hparams,
                        model=self.model,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight
                    )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                if self.alg_name == 'IKE':
                    if self.hparams.k != 0:
                        metrics = {
                            'case_id': i,
                            "time": exec_time,
                            "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, icl_examples,
                                                                request, self.hparams.device),
                            "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, [''],
                                                                request, self.hparams.device, pre_edit=True)
                        }
                    else:
                        from copy import deepcopy
                        prompt_new_request = deepcopy(request)
                        prefix_template = self.hparams.template.format(prompt=request["prompt"],target=request["target"])
                        prompt_new_request["prompt"] = prefix_template + request["prompt"]
                        prompt_new_request["rephrase_prompt"] = prefix_template + request["rephrase_prompt"]
                        prompt_new_request["locality_prompt"] = prefix_template + request["locality_prompt"]
                        prompt_new_request["multimodal_locality_prompt"] = prefix_template + request["multimodal_locality_prompt"]
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
                                                                prompt_new_request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
                                                                request, self.hparams.device)
                
                        }
                else:
                    tqdm.write("Collecting post-editing results...")
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "pre": pre_edit_result,
                        "post":  compute_lalm_hf_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)
                    }
                    
                    # print("=" * 5 + " Evaluating pre-editing metrics... " + "=" * 5)
                    # from desta import DeSTA25AudioModel
                    
                    # self.model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", cache_dir=self.hparams.cache_dir, device_map="auto")
                    # self.tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct", cache_dir=self.hparams.cache_dir)
                    # metrics["pre"] = compute_lalm_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)
                    
                    # print("Evaluating post-editing metrics...")
                    # metrics["post"] = compute_lalm_hf_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)
                    
                    
                # TODO: what is the definition of correct in out locality?
                loc_keys = [key for key in metrics['post'].keys() if key.startswith("locality") or key.startswith("multimodal_locality")]
                for key in loc_keys:
                    assert len(metrics['post'][key]) == len(metrics['pre'][key])
                    base_logits = torch.tensor(metrics['pre'][key]).to(torch.float32)
                    post_logits = torch.tensor(metrics['post'][key]).to(torch.float32)
                    metrics['post'][f'{key}_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
                     
                    
                # if 'locality_output' in metrics['post'].keys():
                #     assert len(metrics['post']['locality_output']) == \
                #             len(metrics['pre']['locality_output'])
                #     base_logits = torch.tensor(metrics['pre']['locality_output']).to(torch.float32)
                #     post_logits = torch.tensor(metrics['post']['locality_output']).to(torch.float32)
                #     metrics['post']['locality_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
                #     metrics['post'].pop('locality_output')
                #     metrics['pre'].pop('locality_output')
                    
                # if 'multimodal_locality_output' in metrics['post'].keys():
                #     assert len(metrics['post']['multimodal_locality_output']) == \
                #             len(metrics['pre']['multimodal_locality_output'])
                #     base_image_logits = torch.tensor(metrics['pre']['multimodal_locality_output']).to(torch.float32)
                #     post_image_logits = torch.tensor(metrics['post']['multimodal_locality_output']).to(torch.float32)
                #     metrics['post']['multimodal_locality_acc'] = sum(post_image_logits.view(-1) == base_image_logits.view(-1))/post_image_logits.view(-1).shape[0]
                #     metrics['post'].pop('multimodal_locality_output')
                #     metrics['pre'].pop('multimodal_locality_output')


                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    # LOG.info(
                    #     f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    # )
                    LOG.info(
                        f"{i} editing: {request}  \n {metrics}"
                    )

                all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy

    def _decode(self, inputs: Union[List[int], torch.Tensor],
                skip_special_tokens: bool = True):
        return self.tokenizer.batch_decode(
            inputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False
        )

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
                    
    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }
        
        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
            
        return data
    