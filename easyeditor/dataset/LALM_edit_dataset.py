"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_lalm_dataset import BaseDataset
from ..trainer.utils import dict_to
import random
import typing
import torch
import transformers
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

class Qwen2AudioDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, cache_dir=None, *args, **kwargs):
        # get processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=config.cache_dir)


        audio_root = config.audio_image
        super().__init__(audio_root, [data_dir])

        self.config = config
        self.max_length = 256

        # self.prompt = "Question: {} Short answer:"
        if size is not None:
            self.data = self.data[:size]  

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def create_message(self, sample):
        """
        sample: {
            "audio_path": "path/to/audio/file",
            "question": "What is the question?",
            "answer": "What is the answer?"
        }
        """
        message = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": sample["audio_path"]},
                {"type": "text", "text": sample["question"]}
            ]}
        ]
        return message

    def collect_audio_from_messages(self, messages):
        audios = []
        for conversation in messages:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'], 
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )
        return audios

    def collate_fn(self, batch):
        # src = [b['reliability_question'] for b in batch]
        # trg = [" " + b['reliability_answer'] for b in batch]
        # rephrase = [b['rephrase_prompt'] for b in batch]
        # image = [b['image'] for b in batch]
        # image_rephrase = [b['image_rephrase'] for b in batch]
        # loc_q = [b["locality_prompt"] for b in batch]
        # loc_a = [" " + b["locality_ground_truth"] for b in batch]
        # m_loc_image = [b['multimodal_locality_image'] for b in batch]
        # m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        # m_loc_a = [" " + b['multimodal_locality_ground_truth'] for b in batch]
        
        # edit_inner (reliability)
        reliability_prompts = [self.create_message(b['reliability']) for b in batch] # We currently apply the chat template to the reliability prompts
        reliability_audios = self.collect_audio_from_messages(reliability_prompts)
        reliability_target = [b['reliability']['reliability_answer'] for b in batch]
        
        edit_inner = {}
        # edit_inner['image'] = torch.stack(image, dim=0)
        # edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        # edit_inner['labels'] = trg
        # if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
        #     edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
        #     edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        # else:
        #     edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
        #     edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        reliability_inputs = self.processor(
            audio=reliability_audios,
            text=reliability_prompts,
            return_tensors="pt",
            padding=True
        )
        
        # edit_outer (generality)
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [self.prompt.format(r) + t for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r), add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r))) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [q + a for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
        
        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]

        # cond
        # cond = self.tok(
        #     cond,
        #     return_tensors="pt",
        #     padding=True,
        #     max_length=self.max_length,
        #     truncation=True,
        # ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            # "cond": cond
        }
        return dict_to(batch, self.config.device)
