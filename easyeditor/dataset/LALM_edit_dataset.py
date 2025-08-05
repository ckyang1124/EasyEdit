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
from transformers import AutoProcessor, AutoTokenizer
import librosa
from desta.utils.audio import AudioSegment
from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions

class Qwen2AudioDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, cache_dir=None, *args, **kwargs):
        # get processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir=config.cache_dir)


        audio_root = config.audio_root
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
    
    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

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

        if len(audios) > 0:
            inputs = self.processor(
                audios=audios,
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
        
        labels = self.get_edit_labels(labels)  # Mask padding tokens with -100 for loss calculation
        
        edit = inputs
        edit['labels'] = labels
        
        # Currently do not include prompts_len as the purpose of this is not clear
        # edit['prompts_len'] = [len(self.processor.tokenizer.encode(src, add_special_tokens=False)) for src in prompts_chat_template]

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



class DeSTA25AudioDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, cache_dir=None, *args, **kwargs):
        
        # Set up tokenizer
        self.audio_locator = "<|AUDIO|>"
        self.placeholder_token = "<|reserved_special_token_87|>"
        self.tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct", cache_dir=config.cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_tokens([self.audio_locator])
        
        # Set up processor
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", cache_dir=config.cache_dir)
        self.prompt_size = 64 # TODO: Check whether this is the right size for DeSTA2.5
        
        audio_root = config.audio_root
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
    
    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
    
    def create_message(self, sample):
        """
        sample: {
            "audio_path": "path/to/audio/file" or None,
            "question": "What is the question?",
            "answer": "What is the answer?",
            "transcription": "What is the transcription?" or None
        }
        """
        if sample['audio_path'] is not None:
            message = [
                # Uncomment the following line if you want to add a system message
                # {
                #     "role": "system",
                #     "content": "Focus on the audio clips and instructions."
                # },
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{sample['question']}",
                    "audios": [{
                        "audio": sample['audio_path'],
                        "text": sample['transcription']
                    }]
                }
            ]
        else:
            message = [
                # Uncomment the following line if you want to add a system message
                # {
                #     "role": "system",
                #     "content": "Focus on the audio clips and instructions."
                # },
                {
                    "role": "user",
                    "content": f"{sample['question']}"
                }
            ]

        return message
    
    def collect_audio_and_transcription_from_messages(self, messages_list):
        all_audios = []
        all_transcriptions = []
        for messages in messages_list:
            for message in messages:
                content = message["content"]
                audios = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), "audio count does not match (<|AUDIO|>) count"

                for audio in audios:
                    # if audio['audio'] is not None:
                        all_audios.append(audio["audio"])
                        all_transcriptions.append(audio.get("text"))

        return all_audios, all_transcriptions
    
    def process_and_tokenize_batch(self, batch, key='reliability'):
        prompts = [self.create_message(b[key]) for b in batch]
        targets = [b[key]['answer'] for b in batch]
        all_audios, all_transcriptions = self.collect_audio_and_transcription_from_messages(prompts)

        if len(all_audios) > 0:
            batch_features = []
            for i, (audio, trans) in enumerate(zip(all_audios, all_transcriptions)):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file {audio} does not exist.")

                # Extract audio features
                feature = AudioSegment.from_file(
                    audio,
                    target_sr=16000,
                    channel_selector="average"
                ).samples

                batch_features.append(feature)
            
            batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
            # batch_features = batch_features.to(self.device)
            audio_size_list = [self.prompt_size] * len(batch_features)
                    
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in all_transcriptions
            ]


            audio_context_list = []
            start_positions_list = []
            for messages, trg in zip(prompts, targets):
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ) + trg # Concat with target

                # <start_audio><|AUDIO|><end_audio> is a indicator used in the training stage
                # We replace <|AUDIO|> with <start_audio><|AUDIO|><end_audio> here
                audio_context = audio_context.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>")

                audio_context, start_positions = _prepare_audio_context_and_start_positions(
                        token_list=self.tokenizer.tokenize(audio_context), 
                        audio_locator=self.audio_locator,
                        audio_size_list=audio_size_list,
                        transcription_size_list=transcription_size_list,
                        placeholder_token=self.placeholder_token
                    )


                audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
                audio_context_list.append(audio_context)
                start_positions_list.append(start_positions)
            
            audio_context_inputs = self.tokenizer(
                audio_context_list,
                truncation=True,
                # padding="longest",
                return_tensors="pt",
                padding=True,
                max_length=self.max_length, # We set a max length
                return_length=True,
                add_special_tokens=False,
            )

            audio_context_batch_start_positions = []
            for i in range(audio_context_inputs["length"].size(0)):
                total_length = audio_context_inputs["length"][i]
                pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()

                for start_position in start_positions_list[i]:
                    audio_context_batch_start_positions.append((i, start_position + pad_length))

            batch_transcription_ids = []
            for transcription in all_transcriptions:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long()
                )

            inputs = {
                "batch_features": batch_features,
                "batch_transcription_ids": batch_transcription_ids,
                "input_ids": audio_context_inputs["input_ids"],
                "attention_mask": audio_context_inputs['attention_mask'],
                "batch_start_positions": audio_context_batch_start_positions,
            }

        else:
            inputs = self.tokenizer.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = [src + trg for (src, trg) in zip(inputs, targets)]  # Concat with target
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, max_length=self.max_length)
            inputs['batch_features'] = None
            inputs['batch_transcription_ids'] = None
            inputs['batch_start_positions'] = None
            
        labels = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )["input_ids"]
        
        labels = self.get_edit_labels(labels)  # Mask padding tokens with -100 for loss calculation
        inputs['labels'] = labels
        
        return inputs
    
    def collate_fn(self, batch):
        keys = batch[0].keys()
        collated_batch = {
            key: self.process_and_tokenize_batch(batch, key)
            for key in keys
        }

        return dict_to(collated_batch, self.config.device)
            

### test case for debugging
if __name__ == "__main__":
    import types
    from torch.utils.data import DataLoader
    
    data_dir = "/work/b08202033/lalm-knowledge-editing/metadata/Animal.json"
    config = types.SimpleNamespace()
    config.cache_dir = "/work/b08202033/SLLM_multihop/cache"
    config.audio_root = "/work/b08202033/lalm-knowledge-editing/audio_data"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # qwen_dataset = Qwen2AudioDataset(data_dir, size=10, cache_dir=config.cache_dir)
    desta_dataset = DeSTA25AudioDataset(data_dir, size=10, cache_dir=config.cache_dir)
    # loader = DataLoader(qwen_dataset, batch_size=2, collate_fn=qwen_dataset.collate_fn)
    loader = DataLoader(desta_dataset, batch_size=2, collate_fn=desta_dataset.collate_fn)

    for batch in loader:
        for key in batch:
            for q in batch[key].keys():
                print(f"{key} - {q}: {batch[key][q].shape if isinstance(batch[key][q], torch.Tensor) else batch[key][q]}")
        break

    




