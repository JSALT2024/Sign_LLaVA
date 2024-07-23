#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import glob

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN, DEFAULT_VIDEO_TOKEN


def load_pretrained_model(config, use_flash_attn=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    lora_config = config['LoraArguments']
    if lora_config['bits']==8:
        kwargs['load_in_8bit'] = True
    elif lora_config['bits']==4:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16 if config['GenerateArguments']['bf16'] else torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    model_path = config['GenerateArguments']['model_path']
    model_base = config['GenerateArguments']['model_base']
    checkpoint_num = config['GenerateArguments']['checkpoint_num']
    # Load LLaVA model
    if lora_config['lora_enable']:
        from llava.model.language_model.llava_llama import LlavaConfig
        lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        if tokenizer.unk_token is None:
            tokenizer.add_special_tokens({"unk_token":"<unk>"})
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        print('Loading LLaVA from base model...')
        model = SignLlavaLlamaForCausalLM.from_pretrained(
                model_base,
                local_files_only=True,
                low_cpu_mem_usage=True, 
                config=lora_cfg_pretrained, 
                sign_model_args=config['SignModelArguments'],
                sign_data_args=config['SignDataArguments'],
                **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, f"checkpoint-{checkpoint_num}", 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            non_lora_trainables = {}
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        peft_model_path = os.path.join(model_path, f"checkpoint-{checkpoint_num}")
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, peft_model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    else:
        # this may be mm projector only
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        if tokenizer.unk_token is None:
            tokenizer.add_special_tokens({"unk_token":"<unk>"})
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = SignLlavaLlamaForCausalLM.from_pretrained(
                model_base,
                local_files_only=True,
                low_cpu_mem_usage=True, 
                sign_model_args=config['SignModelArguments'],
                sign_data_args=config['SignDataArguments'],
                **kwargs)
        tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        model_state_path = glob.glob(os.path.join(model_path, f"checkpoint-{checkpoint_num}/global_step{checkpoint_num}/*model_states.pt"))[0]
        mm_projector_weights = torch.load(model_state_path, map_location='cpu')['module']
        mm_projector_weights = {k: v.to(kwargs['torch_dtype']) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
