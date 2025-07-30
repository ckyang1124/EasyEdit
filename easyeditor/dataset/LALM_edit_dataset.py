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
from transformers import AutoProcessor
import librosa

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
            "audio_path": "path/to/audio/file" or None,
            "question": "What is the question?",
            "answer": "What is the answer?"
        }
        """
        message = [
            {
                "role": "user", 
                "content": (
                    ([{"type": "audio", "audio_url": sample["audio_path"]}] if sample["audio_path"] is not None else [])
                    + [{"type": "text", "text": sample["question"]}]
                )
            }
        ]
        return message

    def collect_audio_from_messages(self, messages):
        audios = []
        for conversation in messages:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            if ele['audio_url'] is not None:
                                # None for no audio
                                audios.append(
                                    librosa.load(
                                        ele['audio_url'], 
                                        sr=self.processor.feature_extractor.sampling_rate)[0]
                                )
        return audios

    def process_and_tokenize_batch(self, batch, key='reliability'):
        prompts = [self.create_message(b[key]) for b in batch]
        audios = self.collect_audio_from_messages(prompts)
        target = [b[key]['answer'] for b in batch]
        prompts_chat_template = [self.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in prompts] # Only question
        input_text = [src + trg for (src, trg) in zip(prompts_chat_template, target)] # Concat question with labels

        if len(audios) != 0:
            inputs = self.processor(
                audio=audios,
                text=input_text,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True
            )
        else:
            inputs = self.processor.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True
            )
            inputs['input_features'] = None
            inputs['feature_attention_mask'] = None
            
        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )["input_ids"]
        
        edit = inputs
        edit['labels'] = labels
        edit['prompts_len'] = [len(self.tok.encode(src, add_special_tokens=False)) for src in prompts_chat_template]

        return edit

    def collate_fn(self, batch):
        keys = batch[0].keys()
        collated_batch = {
            key: self.process_and_tokenize_batch(batch, key)
            for key in keys
        }
        

        # cond (currently skipped as I am not sure the purpose of this)
        # cond = self.tok(
        #     cond,
        #     return_tensors="pt",
        #     padding=True,
        #     max_length=self.max_length,
        #     truncation=True,
        # ).to(self.config.device)

        return dict_to(collated_batch, self.config.device)






