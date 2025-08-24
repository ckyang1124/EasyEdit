from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from ..dataset.processor.llavaov_processors import LLaVAOneVisionProcessor
from ..dataset.processor.qwen2vl_processors import Qwen2VLProcessor
from .editor import BaseEditor
import os.path
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

from ..trainer.utils import dict_to


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
        self.audio_root = hparams.audio_root 
        self.max_new_tokens = 256

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
    
    # def _process_one_sample(self, edit_sample: dict, track: str):
    #     """
    #     edit_sample should be the format in our dataset. 
        
    #     Return a dict with the following structure:
    #     {
    #         "...": {
    #             "audio_path": str,
    #             "question": str,
    #             "transcription": str,
    #             "answer": str,
    #         }
    #     }
    #     """
        
    #     item = {}
    #     item["reliability"] = {
    #         'audio_path': os.path.join(self.audio_root, track, edit_sample["file"]),
    #         'question': edit_sample["reliability_question"],
    #         'answer': edit_sample["edited_answer"],
    #         'transcription': edit_sample["transcription"],
    #     }
    #     for i, gen in enumerate(edit_sample["generality"]):
    #         item[f"generality_type_{i}"] = {
    #             'audio_path': os.path.join(self.audio_root, track, edit_sample["file"]),
    #             'question': gen["question"],
    #             'answer': gen["answer"],
    #             'transcription': edit_sample["transcription"],
    #         }
    #     for i, loc in enumerate(edit_sample["locality"]["audio"]):
    #         ti = i
    #         if track.lower() == "gender":
    #             # no audio locality type 1, so we skip it
    #             if i >= 1:
    #                 ti = i + 1
    #         item[f"locality_audio_type_{i}"] = {
    #             'audio_path': os.path.join(self.audio_root, loc["track"], loc["file"]),
    #             'question': loc["question"],
    #             'answer': loc["answer"],
    #             'transcription': edit_sample["transcription"],
    #         }
    #     item["locality_text"] = {
    #         'question': edit_sample["locality"]["text"]["question"],
    #         'answer': edit_sample["locality"]["text"]["answer"],
    #     }
    #     item["portability"] = {
    #         'audio_path': os.path.join(self.audio_root, track, edit_sample["portability"]["audio"]["file"]),
    #         'question': edit_sample["portability"]["audio"]["question"],
    #         'answer': edit_sample["portability"]["audio"]["answer"],
    #         'transcription': edit_sample["transcription"],
    #     }
    #     return item
    
    # def _prepare_message(self, request: dict) -> str:
    #     """
    #     Prepare the message for the model.
    #     request should be:
    #     {
    #         "audio_path": str,
    #         "question": str,
    #         "transcription": str,
    #         "answer": str,
    #     }
    #     """
    #     if "qwen2-audio" in self.model_name.lower():
    #         message = [
    #             {
    #                 "role": "user", 
    #                 "content": (
    #                     ([{"type": "audio", "audio_url": request["audio_path"]}] if request["audio_path"] is not None else [])
    #                     + [{"type": "text", "text": request["question"]}]
    #                 )
    #             }
    #         ]
    #         return message
    #     elif "desta" in self.model_name.lower():
    #         if request['audio_path'] is not None:
    #             message = [
    #                 {
    #                     "role": "user",
    #                     "content": f"<|AUDIO|>\n{request['question']}",
    #                     "audios": [{
    #                         "audio": request['audio_path'],
    #                         "text": request['transcription']
    #                     }] if request['audio_path'] is not None else []
    #                 }
    #             ]
    #         else:
    #             message = [
    #                 {
    #                     "role": "user",
    #                     "content": f"{request['question']}"
    #                 }
    #             ]

    #         return message
    #     else:
    #         raise NotImplementedError(f"Model {self.model_name} is not supported yet for message preparation.")
    
    # def generate_response(self, request: dict) -> str:
    #     """
    #     request should be:
    #     {
    #         "audio_path": str,
    #         "question": str,
    #         "transcription": str,
    #         "answer": str,
    #     }
    #     """
    #     messages = self._prepare_message(request)
    #     if "qwen2-audio" in self.model_name.lower():
    #         text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #         audios = []
    #         for message in messages:
    #             if isinstance(message["content"], list):
    #                 for ele in message["content"]:
    #                     if ele["type"] == "audio":
    #                         if ele['audio_url'] is not None:
    #                             # None for no audio
    #                             audios.append(
    #                                 librosa.load(
    #                                     ele['audio_url'], 
    #                                     sr=self.processor.feature_extractor.sampling_rate)[0]
    #                             )

    #         inputs = self.processor(
    #             text=text, 
    #             audios=audios, 
    #             return_tensors="pt", 
    #             padding=True,
    #             sampling_rate=self.processor.feature_extractor.sampling_rate
    #         )
    #         inputs.input_ids = inputs.input_ids.to("cuda")
    #         generate_ids = self.model.generate(
    #             **inputs, 
    #             max_length=self.max_new_tokens, 
    #             do_sample=False, 
    #             top_p=1.0, 
    #             temperature=1.0
    #         )
    #         generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    #         response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #         return response

    #     elif "desta" in self.model_name.lower():
    #         outputs = self.model.generate(
    #             messages=messages,
    #             do_sample=False,
    #             top_p=1.0,
    #             temperature=1.0,
    #             max_new_tokens=self.max_new_tokens,
    #         )
    #         return outputs.text
    
    def single_edit_dataset(self, ds: Union[DeSTA25AudioDataset, Qwen2AudioDataset], output_path: str, generate_pre_edit: bool = True) -> None:
        LOG.info(f"Start editing the dataset and save results to {output_path}...")
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
            
        
    # def single_edit(self, metadata_path: str, track: str, output_path: str):
    #     data = json.load(open(metadata_path, 'r'))
    #     all_results = []
    #     for edit_sample in tqdm(data, desc='Test Editing samples', dynamic_ncols=True):
    #         pre_edit = {}
    #         formatted_sample = self._process_one_sample(edit_sample, track)
    #         for key, request in formatted_sample.items():
    #             response = self.generate_response(request)
    #             pre_edit[key] = response
            
    #         # perform the editing
    #         # TODO: add the editing logic here
            
    #         post_edit = {}
    #         for key, request in formatted_sample.items():
    #             response = self.generate_response(request)
    #             post_edit[key] = response
            
    #         all_results.append({
    #             **edit_sample,
    #             "pre_edit": pre_edit,
    #             "post_edit": post_edit
    #         })
    #     with open(output_path, "w") as f:
    #         json.dump(all_results, f, indent=4, ensure_ascii=False)
        
            
    # def edit(self,
    #         # prompts: Union[str, List[str]],
    #         # targets: Union[str, List[str]],
    #         # image: Union[str, List[str]],
    #         # file_type: Union[str,List[str]] = None,
    #         # rephrase_prompts: Optional[Union[str, List[str]]] = None,
    #         # rephrase_image: Optional[Union[str, List[str]]] = None,
    #         # locality_inputs: Optional[dict] = None,
    #         # edit_sample: Union[dict, List[dict]],
    #         testing_ds: Dataset,
    #         edit_sample: Union[dict, List[dict]] = None,
    #         keep_original_weight=False,
    #         verbose=True,
    #         sequential_edit=False,
    #         **kwargs
    #         ):
    #     """
    #     edit_sample: dict or list of dict {
    #         "audio": 
    #         "transcription":
    #         "reliability_question":
    #         "original_answer":
    #         "edited_answer":
    #         "generality": []
    #         "locality": {
    #             "audio": [],
    #             "text": [],
    #         }
    #         "portability": {
    #             "audio": {}
    #         }
    #     }
    #     `prompts`: list or str
    #         the prompts to edit
    #     `targets`: str
    #         the expected outputs
    #     `image`: dict
    #         for multimodal
    #     """
        
    #     if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
    #         # TODO: I am not sure what is it for
    #         self.hparams.batch_size = 1
        
    #     assert testing_ds or edit_sample, \
    #         'You must provide either testing_ds or edit_sample to edit.'
    #     if testing_ds:
    #         assert edit_sample is None, 'You can not set both testing_ds and editing_sample at the same time.'
    #         test_loader = DataLoader(testing_ds, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=testing_ds.collate_fn)
    #     if edit_sample:
    #         # TODO
    #         raise NotImplementedError("Providing edit_sample directly is not supported yet. Please use testing_ds instead.")
    #         # assert testing_ds is None, 'You can not set both testing_ds and editing_sample at the same time.'
    #     if sequential_edit:
    #         # TODO
    #         raise NotImplementedError("Sequential editing is not supported yet. Please set sequential_edit=False.")
            
    #     # sequential editing
    #     if sequential_edit:
    #         # TODO: figure out what to do when sequential_edit is True
    #         for i, request in enumerate(tqdm(requests, total=len(requests))):
    #             if self.alg_name == 'IKE' and self.hparams.k == 0:
    #                 edited_model = self.model
    #                 weights_copy = None
    #             else:
    #                 edited_model, weights_copy = self.apply_algo(
    #                     self.model,
    #                     self.tokenizer,
    #                     request,
    #                     self.hparams,
    #                     copy=False,
    #                     return_orig_weights=True,
    #                     keep_original_weight=keep_original_weight,
    #                     train_ds=None
    #                 )
    #         exec_time = time() - start
    #         if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
    #             print("Start saving the WISE model!")
    #             edited_model.save(self.hparams.save_path)
                
    #         all_metrics = []
    #         exec_time = time() - start
    #         for i, request in enumerate(tqdm(requests, total=len(requests))):
    #             if self.alg_name == 'IKE':
    #                 if self.hparams.k != 0:    
    #                     metrics = {
    #                         'case_nums': i,
    #                         "time": exec_time,
    #                         "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, icl_examples,
    #                                                             request, self.hparams.device),
    #                         "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, [''],
    #                                                             request, self.hparams.device, pre_edit=True)
    #                     }
    #                 else:
    #                     # QUESTION = request["prompt"]
    #                     # ANSWER = request["target"]
    #                     from copy import deepcopy
    #                     prompt_new_request = deepcopy(request)
    #                     prefix_template = self.hparams.template.format(prompt=request["prompt"],target=request["target"])
    #                     prompt_new_request["prompt"] = prefix_template + request["prompt"]
    #                     prompt_new_request["rephrase_prompt"] = prefix_template + request["rephrase_prompt"]
    #                     prompt_new_request["locality_prompt"] = prefix_template + request["locality_prompt"]
    #                     prompt_new_request["multimodal_locality_prompt"] = prefix_template + request["multimodal_locality_prompt"]
    #                     metrics = {
    #                         'case_nums': i,
    #                         "time": exec_time,
    #                         "post": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
    #                                                             prompt_new_request, self.hparams.device),
    #                         "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
    #                                                             request, self.hparams.device)
    #                     }
    #             else:
    #                 if self.model_name in ['minigpt4', 'blip2']:
    #                     metrics = {
    #                         'case_nums': i,
    #                         "time": exec_time,
    #                         "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer,
    #                                                             request, self.hparams.device),
    #                         "pre": compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
    #                                                             request, self.hparams.device)
    #                     }
    #                 elif self.model_name in ['llava-onevision', 'qwen2-vl']:
    #                     metrics = {
    #                         'case_nums': i,
    #                         "time": exec_time,
    #                         "post": compute_multimodal_hf_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer,
    #                                                             request, self.hparams.device),
    #                         "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
    #                                                             request, self.hparams.device)
    #                     }
                                
    #             if 'locality_output' in metrics['post'].keys():
    #                 assert len(metrics['post']['locality_output']) == \
    #                         len(metrics['pre']['locality_output'])
    #                 base_logits = torch.tensor(metrics['pre']['locality_output']).to(torch.float32)
    #                 post_logits = torch.tensor(metrics['post']['locality_output']).to(torch.float32)
    #                 metrics['post']['locality_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
    #                 metrics['post'].pop('locality_output')
    #                 metrics['pre'].pop('locality_output')
                    
    #             if 'multimodal_locality_output' in metrics['post'].keys():
    #                 assert len(metrics['post']['multimodal_locality_output']) == \
    #                         len(metrics['pre']['multimodal_locality_output'])
    #                 base_image_logits = torch.tensor(metrics['pre']['multimodal_locality_output']).to(torch.float32)
    #                 post_image_logits = torch.tensor(metrics['post']['multimodal_locality_output']).to(torch.float32)
    #                 metrics['post']['multimodal_locality_acc'] = sum(post_image_logits.view(-1) == base_image_logits.view(-1))/post_image_logits.view(-1).shape[0]
    #                 metrics['post'].pop('multimodal_locality_output')
    #                 metrics['pre'].pop('multimodal_locality_output')

    #             LOG.info(f"Evaluation took {time() - start}")
    #             all_metrics.append(metrics)
        
    #     # single editing
    #     else:
    #         all_metrics = []
    #         for i, request in enumerate(tqdm(test_loader, disable=not verbose, desc='Editing dataset', dynamic_ncols=True)):
    #             if i > 1:
    #                 break
                
    #             tqdm.write("Collecting pre-editing results...")
    #             pre_edit_result = compute_lalm_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)    
                
    #             start = time()
    #             if self.alg_name == 'IKE':
    #                 # TODO: modify IKE for LALM in the future
    #                 # What is self.hparams.k?
    #                 # Is self.hparams.k the number of in-context examples?
    #                 if self.hparams.k != 0:                    
    #                     assert 'train_ds' in kwargs.keys(), 'IKE need train_ds (For getting In-Context prompt)'
    #                     edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
    #                         self.model,
    #                         self.tokenizer,
    #                         request,
    #                         self.hparams,
    #                         copy=False,
    #                         return_orig_weights=True,
    #                         keep_original_weight=keep_original_weight,
    #                         train_ds=kwargs['train_ds']
    #                     )
    #                 else:
    #                     edited_model = self.model
    #                     weights_copy = None
    #             else:
    #                 edited_model, weights_copy = self.apply_algo(
    #                     # self.tokenizer,
    #                     requests=request["reliability"], # [request],
    #                     hparams=self.hparams,
    #                     model=self.model,
    #                     copy=False,
    #                     return_orig_weights=True,
    #                     keep_original_weight=keep_original_weight
    #                 )
    #             exec_time = time() - start
    #             LOG.info(f"Execution {i} editing took {exec_time}")
    #             start = time()
    #             if self.alg_name == 'IKE':
    #                 if self.hparams.k != 0:
    #                     metrics = {
    #                         'case_id': i,
    #                         "time": exec_time,
    #                         "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, icl_examples,
    #                                                             request, self.hparams.device),
    #                         "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tokenizer, [''],
    #                                                             request, self.hparams.device, pre_edit=True)
    #                     }
    #                 else:
    #                     from copy import deepcopy
    #                     prompt_new_request = deepcopy(request)
    #                     prefix_template = self.hparams.template.format(prompt=request["prompt"],target=request["target"])
    #                     prompt_new_request["prompt"] = prefix_template + request["prompt"]
    #                     prompt_new_request["rephrase_prompt"] = prefix_template + request["rephrase_prompt"]
    #                     prompt_new_request["locality_prompt"] = prefix_template + request["locality_prompt"]
    #                     prompt_new_request["multimodal_locality_prompt"] = prefix_template + request["multimodal_locality_prompt"]
    #                     metrics = {
    #                         'case_nums': i,
    #                         "time": exec_time,
    #                         "post": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
    #                                                             prompt_new_request, self.hparams.device),
    #                         "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer,
    #                                                             request, self.hparams.device)
                
    #                     }
    #             else:
    #                 tqdm.write("Collecting post-editing results...")
    #                 metrics = {
    #                     'case_id': i,
    #                     "time": exec_time,
    #                     "pre": pre_edit_result,
    #                     "post":  compute_lalm_hf_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)
    #                 }
                    
    #                 # print("=" * 5 + " Evaluating pre-editing metrics... " + "=" * 5)
    #                 # from desta import DeSTA25AudioModel
                    
    #                 # self.model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", cache_dir=self.hparams.cache_dir, device_map="auto")
    #                 # self.tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct", cache_dir=self.hparams.cache_dir)
    #                 # metrics["pre"] = compute_lalm_hf_edit_results(self.model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)
                    
    #                 # print("Evaluating post-editing metrics...")
    #                 # metrics["post"] = compute_lalm_hf_edit_results(edited_model, self.model_name, self.hparams, self.tokenizer, request, self.hparams.device, self._decode)
                    
                    
    #             # TODO: what is the definition of correct in out locality?
    #             loc_keys = [key for key in metrics['post'].keys() if key.startswith("locality") or key.startswith("multimodal_locality")]
    #             for key in loc_keys:
    #                 assert len(metrics['post'][key]) == len(metrics['pre'][key])
    #                 base_logits = torch.tensor(metrics['pre'][key]).to(torch.float32)
    #                 post_logits = torch.tensor(metrics['post'][key]).to(torch.float32)
    #                 metrics['post'][f'{key}_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
                     
                    
    #             # if 'locality_output' in metrics['post'].keys():
    #             #     assert len(metrics['post']['locality_output']) == \
    #             #             len(metrics['pre']['locality_output'])
    #             #     base_logits = torch.tensor(metrics['pre']['locality_output']).to(torch.float32)
    #             #     post_logits = torch.tensor(metrics['post']['locality_output']).to(torch.float32)
    #             #     metrics['post']['locality_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
    #             #     metrics['post'].pop('locality_output')
    #             #     metrics['pre'].pop('locality_output')
                    
    #             # if 'multimodal_locality_output' in metrics['post'].keys():
    #             #     assert len(metrics['post']['multimodal_locality_output']) == \
    #             #             len(metrics['pre']['multimodal_locality_output'])
    #             #     base_image_logits = torch.tensor(metrics['pre']['multimodal_locality_output']).to(torch.float32)
    #             #     post_image_logits = torch.tensor(metrics['post']['multimodal_locality_output']).to(torch.float32)
    #             #     metrics['post']['multimodal_locality_acc'] = sum(post_image_logits.view(-1) == base_image_logits.view(-1))/post_image_logits.view(-1).shape[0]
    #             #     metrics['post'].pop('multimodal_locality_output')
    #             #     metrics['pre'].pop('multimodal_locality_output')


    #             LOG.info(f"Evaluation took {time() - start}")

    #             if verbose:
    #                 # LOG.info(
    #                 #     f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
    #                 # )
    #                 # LOG.info(
    #                 #     f"{i} editing: {request}  \n {metrics}"
    #                 # )
    #                 pass

    #             all_metrics.append(metrics)

    #     return all_metrics, edited_model, weights_copy

    def _decode(self, inputs: Union[List[int], torch.Tensor],
                skip_special_tokens: bool = True):
        return self.tokenizer.batch_decode(
            inputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False
        )