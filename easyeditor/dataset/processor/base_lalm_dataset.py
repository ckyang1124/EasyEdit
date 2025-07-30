"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
import os
            
class BaseDataset(Dataset):
    def __init__(
        self, audio_root=None, metadata_path=[]
    ):
        """
        audio_root (string): Root directory of audio files (e.g. audio_data/Gender/)
        metadata_path (string): Path to the metadata file, which is expected to be a JSON file
        """
        self.audio_root = audio_root
        self.metadata_path = metadata_path

        self.data = []
        tracks = ['Gender', 'Language', 'Emotion', 'Animal']
        
        for metadata in metadata_path:
            for t in tracks:
                if t in metadata:
                    track = t
                    break

            for data_point, data_content in json.load(open(metadata, "r")).items():                
                # Reliability
                item = {}
                
                reliability_audio_path = os.path.join(self.audio_root, track, data_point)
                assert os.path.exists(reliability_audio_path), f"Audio file {reliability_audio_path} does not exist."
                reliability_question = data_content['reliability_question']
                reliability_answer = data_content['edited_answer']
                reliability_data = {
                    'audio_path': reliability_audio_path,
                    'question': reliability_question,
                    'answer': reliability_answer
                }
                item['reliability'] = reliability_data
                
                # Generality
                generality_data = [self.process(data_content['generality'][i], track=track) for i in range(len(data_content['generality']))]
                for i, data in enumerate(generality_data):
                    item[f'generality_type_{i}'] = data
                
                
                # Locality
                locality_audio_data = [self.process(data_content['locality']['audio'][i], track=track) for i in range(len(data_content['locality']['audio']))]
                for i, data in enumerate(locality_audio_data):
                    item[f'locality_audio_type_{i}'] = data
                    
                # text locality should only have one data per sample
                assert len(data_content['locality']['text']) == 1, "Locality text should only have one data per sample."    
                locality_text_data = [self.process(data_content['locality']['text'][i], track=track) for i in range(len(data_content['locality']['text']))]
                item['locality_text'] = locality_text_data[0]
                
                self.data.append(item)

        self._add_instance_ids()

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        return default_collate(samples)

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.data):
            ann[key] = str(idx)
            
    def process(self, sample, track='Gender'):
        """
        Process the sample from the dataset into a specific format.
        """
        if 'file' in sample:
            if 'track' in sample:
                audio_path = os.path.join(self.audio_root, sample['track'], sample['file'])
                assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
            else:
                audio_path = os.path.join(self.audio_root, track, sample['file'])
                assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
        else:
            audio_path = None
            
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        
        return {
            'audio_path': audio_path,
            'question': question,
            'answer': answer
        }

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
