# Adapted from https://github.com/nicola-decao/KnowledgeEditor/blob/main/src/models/one_shot_learner.py
"""
@inproceedings{decao2020editing,
 title={Editing Factual Knowledge in Language Models},
 author={Nicola De Cao and Wilker Aziz and Ivan Titov},
 booktitle={arXiv pre-print 2104.08164},
 url={https://arxiv.org/abs/2104.08164},
 year={2021},
}
"""

import copy
import logging

import higher
import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, Qwen2AudioForConditionalGeneration, AutoProcessor
from collections import deque
from .editable_model import EditableModel
from ..models import BertClassifier
from ..utils import _inner_params, _logits
from .MEND import monkeypatch as make_functional
from desta import DeSTA25AudioModel

import librosa

LOG = logging.getLogger(__name__)


class EFK(EditableModel):
    def __init__(self, model, config, model_constructor, editor=None):
        super().__init__(model, config, model_constructor)
        
        if not str(self.config.device).startswith('cuda'):
            self.config.device = f'cuda:{self.config.device}'
            
        if editor is None:
            if isinstance(model, BertClassifier):
                embedding = model.model.embeddings.word_embeddings.weight.data
            elif isinstance(model, BartForConditionalGeneration):
                embedding = model.model.shared.weight.data
            elif isinstance(model, T5ForConditionalGeneration):
                embedding = model.shared.weight.data
            elif isinstance(model, Qwen2AudioForConditionalGeneration):
                embedding = model.language_model.model.embed_tokens.weight.data.cpu()
            elif isinstance(model, DeSTA25AudioModel):
                embedding = model.llm_model.model.embed_tokens.weight.data
            else:
                embedding = model.transformer.wte.weight.data

            # Handling special config structure of DeSTA25AudioModel
            vocab_dim = model.config.vocab_size if hasattr(model.config, 'vocab_size') else model.config.llm_config.vocab_size
            
            editor = OneShotLearner(
                model.named_parameters(),
                vocab_dim=vocab_dim,
                include_set=config.model.inner_params,
                embedding_dim=embedding.shape[-1],
                embedding_init=embedding.clone().to(torch.float32),
                max_scale=1,
            )
            
        self.editor = editor
        if self.config.model_parallel:
            self.editor.to(deque(self.model.parameters(), maxlen=1)[0].device)
        
    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(*inputs, **kwargs)
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'qwen2audio' in self.config.model_name.lower():
            outputs = _logits(
                self.model(input_ids=kwargs['input_ids'],  input_features=kwargs['input_features'], attention_mask=kwargs['attention_mask'], feature_attention_mask=kwargs['feature_attention_mask'])
            )
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'desta' in self.config.model_name.lower():
            outputs = _logits(
                self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'], batch_features=kwargs['batch_features'], 
                           batch_transcription_ids=kwargs['batch_transcription_ids'], batch_start_positions=kwargs['batch_start_positions'])
            )
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        else:
            outputs = _logits(self.model(**kwargs))
        return outputs

    def outer_parameters(self):
        return self.editor.parameters()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        model_keys = self.model.state_dict(
            prefix=prefix, keep_vars=keep_vars
        ).keys()  # Remove model params
        for k in model_keys:
            if f'model.{k}' in state_dict:
                del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert not [
            k for k in res.missing_keys if not k.startswith("model.")
        ], "Should only have missing keys for model."

        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def edit(self, batch, condition=None, detach_history=False, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(batch)        
            if not isinstance(outputs, torch.Tensor):
                batch_labels = outputs.labels
                outputs = outputs.logits
            else:
                batch_labels = batch['labels']
            loss = self.edit_loss_fn(self.config, outputs, batch_labels, multimodal=True)["nll"]          
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            if not kwargs:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
            else:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"], **kwargs)["nll"]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            if not kwargs:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
            else:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"], **kwargs)["nll"]
        elif 'baichuan' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"] 
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]            
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]  
        elif 'qwen2audio' in self.config.model_name.lower():
            outputs = _logits(
                self.model(input_ids=batch['input_ids'],  input_features=batch['input_features'], attention_mask=batch['attention_mask'], feature_attention_mask=batch['feature_attention_mask'])
            )
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
        elif 'desta' in self.config.model_name.lower():
            outputs = _logits(
                self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], batch_features=batch['batch_features'], 
                           batch_transcription_ids=batch['batch_transcription_ids'], batch_start_positions=batch['batch_start_positions'])
            )
            
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]         
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]  
        else:
            outputs = _logits(self.model(**batch))
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]

        names = set([n for n, p in self.model.named_parameters()])
        pset = set(self.config.model.inner_params)
        for p in pset:
            assert p in names, f"inner param {p} not in model"

        grads = torch.autograd.grad(
            loss,
            [
                p
                for (n, p) in _inner_params(
                    self.model.named_parameters(), self.config.model.inner_params
                )
            ],
        )

        editor_device = next(self.editor.parameters()).device
        params_dict = self.editor(
            condition["input_ids"].to(editor_device) if condition is not None else batch["input_ids"].to(editor_device),
            condition["attention_mask"].to(editor_device)
            if condition is not None
            else batch["attention_mask"].to(editor_device),
            {
                n: g.to(torch.float32).to(editor_device)
                for (n, g) in zip(self.config.model.inner_params, grads)
            },
        )

        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            if hasattr(edited_model, 'vad_model') and edited_model.vad_model is not None:
                # If the model has a vad_model, we need to backup it and remove it for monkeypatching
                LOG.info("Removing vad_model for monkeypatching (as it is not part of our editable parameters)")
                del edited_model.vad_model
            edited_model = make_functional(edited_model, in_place=True)

        def new_param(n, p):
            if n not in params_dict:
                return p

            if p.shape[0] == params_dict[n].shape[0]:
                return p + params_dict[n]
            else:
                return p + params_dict[n].T

        edited_model.update_params(
            [new_param(n, p) for (n, p) in edited_model.named_parameters()]
        )

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edited_model.state_dict())
            edited_model = new_model

        return (
            EFK(edited_model, self.config, self.model_constructor, editor=self.editor),
            {},
        )


class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):
        if inputs.shape[0] > 1:
            raise RuntimeError("Can only condition on batches of size 1")

        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
        else:
            raise RuntimeError()

        if a.squeeze().shape[0] != grad.shape[0]:
            return (
                self.max_scale
                * conditioner_norm.sigmoid().squeeze()
                * (grad * a.squeeze().T + b.squeeze().T)
            )
        else:
            return (
                self.max_scale
                * conditioner_norm.sigmoid().squeeze()
                * (grad * a.squeeze() + b.squeeze())
            )


class LSTMConditioner(torch.nn.Module):
    def __init__(
        self,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=256,
        output_dim=1024,
        embedding_init=None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_init,
        )
        self.lstm = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = FeedForward(
            input_dim=hidden_dim * 2,
            num_layers=1,
            hidden_dims=[output_dim],
            activations=[torch.nn.Tanh()],
        )

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks))


class OneShotLearner(torch.nn.Module):
    def __init__(
        self,
        # model,
        named_parameters,
        vocab_dim,
        embedding_dim=768,
        hidden_dim=512,
        condition_dim=768,
        include_set={},
        max_scale=1e-3,
        embedding_init=None,
    ):
        super().__init__()

        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                # for n, p in model.named_parameters()
                for n, p in named_parameters
                if n in include_set
            }
        )

        self.condition = LSTMConditioner(
            vocab_dim,
            embedding_dim,
            hidden_dim,
            condition_dim,
            embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)
        return {
            p: self.conditioners[self.param2conditioner_map[p]](
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()
        }


if __name__ == "__main__":
    import types
    import pdb
    
    model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", cache_dir="/work/b08202033/SLLM_multihop/cache").to("cuda")
    # model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir="/work/b08202033/SLLM_multihop/cache", device_map="auto")
    # model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir="/work/b08202033/SLLM_multihop/cache")
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir="/work/b08202033/SLLM_multihop/cache")
    pdb.set_trace()
    
    config = types.SimpleNamespace()
    config.model = types.SimpleNamespace()
    config.model.inner_params = [
        # "multi_modal_projector.linear.weight",
        # "language_model.model.layers.31.mlp.gate_proj.weight",
        # "language_model.model.layers.31.mlp.up_proj.weight",
        # "language_model.model.layers.31.mlp.down_proj.weight"
        "llm_model.model.layers.31.mlp.gate_proj.weight",
        "llm_model.model.layers.31.mlp.up_proj.weight",
        "llm_model.model.layers.31.mlp.down_proj.weight"
    ]
    config.model_name = "DeSTA25AudioModel"
    config.model_class = "DeSTA25AudioModel"
    # config.model_name = "Qwen2AudioForConditionalGeneration"
    # config.model_class = "Qwen2AudioForConditionalGeneration"
    config.model_parallel = True
    config.device = "1"
    
    for n, p in model.named_parameters():
        if n not in config.model.inner_params:
            p.requires_grad = False
        else:
            p.requires_grad = True
    
    efk = EFK(model, config, lambda: copy.deepcopy(model))
    pdb.set_trace()
    
    x = torch.arange(20).view(1, 20).to(model.device) + 1000 # Random labels for testing
    
    # DeSTA testing case
    messages = [
        {
            "role": "system",
            "content": "Focus on the audio clips and instructions."
        },
        {
            "role": "user",
            "content": "<|AUDIO|>\nDescribe this audio.",
            "audios": [{
                "audio": "/work/b08202033/SLLM_multihop/Gender/data/test/en_test_0_common_voice_en_18556.wav",  # Path to your audio file
                "text": None
            }]
        }
    ]
    
    inputs = model.process_before_forward(messages)
    inputs['labels'] = x.to(model.device)


    # Qwen2Audio testing case
    # test_audio = "/work/b08202033/SLLM_multihop/Gender/data/test/en_test_0_common_voice_en_18556.wav"
    # conversation = [
    #     {"role": "user", "content": [
    #         {"type": "audio", "audio_url": test_audio},
    #         {"type": "text", "text": "What is in this audio?"}
    #     ]}
    # ]
    # text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # audios = []
    # for message in conversation:
    #     if isinstance(message["content"], list):
    #         for ele in message["content"]:
    #             if ele["type"] == "audio":
    #                 audios.append(librosa.load(
    #                     ele['audio_url'], 
    #                     sr=processor.feature_extractor.sampling_rate)[0]
    #                 )
                    
    # inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    # inputs.input_ids = inputs.input_ids.to(model.device)
    # inputs['labels'] = x.to(model.device)
    
    # for k, v in inputs.items():
    #     if isinstance(v, list):
    #         inputs[k] = [item.to(model.device) for item in v]
    #     else:
    #         inputs[k] = v.to(model.device)
    
    pdb.set_trace()
    torch.save(efk.state_dict(), "test_state_efk.pt") # Random intialize a testing checkpoint for sanity check
    efk.load_state_dict(torch.load("test_state_efk.pt", weights_only=False)) # weights_only=False to load the model config for torch==2.7.1
    
    
    orig_logits = efk(**inputs)
    edited, _ = efk.edit(inputs)
    post_logits = efk(**inputs)

    orig_param = [
        p
        for (n, p) in efk.model.named_parameters()
        if n == config.model.inner_params[-1]
    ][0]
    edited_param = [
        p
        for (n, p) in edited.model.named_parameters()
        if n == config.model.inner_params[-1]
    ][0]

    print((orig_param - edited_param).abs().max())