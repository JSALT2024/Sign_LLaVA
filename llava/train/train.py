# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import json
import logging
import os
import pathlib
import random
import shutil
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy
import tokenizers
import torch
import transformers
import yaml
from transformers import set_seed

from llava import conversation as conversation_lib
from llava.constants import *
from llava.model.language_model.llava_llama import SignLlavaLlamaForCausalLM
from llava.train.args_utils import ModelArguments, TrainingArguments, prepare_bnb_args
from llava.train.dataset import SignContextDataset
from llava.train.llava_trainer import LLaVATrainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ExtraArguments:
    yaml_args: str = field(default=None,
                           metadata={"help": "Path to YAML config for overriding arguments."})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, skip_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = skip_modules
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_multimodal(
    sources: Sequence[str]
) -> Dict:
    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)
    return sources


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, video_sep_ids, visual_features = tuple([instance[key] for instance in instances]
                                                                  for key in ("input_ids", "labels", "video_sep_ids",
                                                                              "visual_features"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            video_sep_ids=video_sep_ids,
            visual_features=visual_features,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                sign_data_args,
                                sign_multi_task_args,
                                sign_multi_task_eval_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SignContextDataset(
        tokenizer=tokenizer,
        sign_data_args=sign_data_args,
        sign_multi_task_args=sign_multi_task_args,
        sign_multi_task_eval_args=sign_multi_task_eval_args,
        split="train"
    )
    dev_dataset = SignContextDataset(
        tokenizer=tokenizer,
        sign_data_args=sign_data_args,
        sign_multi_task_args=sign_multi_task_args,
        sign_multi_task_eval_args=sign_multi_task_eval_args,
        split="dev"
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                data_collator=data_collator)


def update_arguments(arg_obj, yaml_dict):
    for k, v in yaml_dict.items():
        setattr(arg_obj, k, v)


def set_same_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def train(attn_implementation=None):
    global local_rank
    # load default args
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, ExtraArguments))
    model_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    # update args from yaml config
    with open(extra_args.yaml_args, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
        update_arguments(model_args, yaml_config['ModelArguments'])
        update_arguments(training_args, yaml_config['TrainingArguments'])
    sign_data_args = yaml_config["SignDataArguments"]
    sign_model_args = yaml_config["SignModelArguments"]
    sign_multi_task_args = yaml_config["SignMultiTaskArguments"]
    sign_multi_task_eval_args = yaml_config.get("SignMultiTaskEvalArguments", None)
    output_dir = training_args.output_dir
    if not training_args.resume_from_checkpoint:
        if os.environ.get("SLURM_JOB_ID", None) is not None:
            output_dir += "-" + os.environ["SLURM_JOB_ID"]
        os.makedirs(output_dir, exist_ok=True)
    training_args.run_name = output_dir.split('/')[-1]
    training_args.output_dir = output_dir

    # set seed
    set_same_seed(training_args.seed)

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # skip projectors for quantization
    # projector_names = []
    # for input_type in sign_model_args:
    #     if 'enable_input' in sign_model_args[input_type] and sign_model_args[input_type]['enable_input']:
    #         projector_names.append(f"{input_type}_projector")
    # skip_modules = projector_names + ['lm_head']
    skip_modules = ['lm_head', "model.encoders"]

    # prepare additional args
    bnb_model_from_pretrained_args = prepare_bnb_args(training_args, compute_dtype, skip_modules)

    # build model
    model = SignLlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        # local_files_only=True,
        sign_model_args=sign_model_args,
        sign_data_args=sign_data_args,
        **bnb_model_from_pretrained_args
    )

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # lora skip lm_head and projectors
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, skip_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": "<unk>"})
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # freeze/unfreeze model
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)

    # load pretrained weights and freeze/unfreeze encoders
    for encoder_name, encoder in model.get_model().encoders.items():
        if "checkpoint_path" in sign_model_args[encoder_name]:
            encoder.initialize_model(**sign_model_args[encoder_name])
        encoder.require_grad(not sign_model_args[encoder_name]["freeze"])

        # move to dtype
        if training_args.bits in [4, 8]:
            encoder.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer, sign_model_args=sign_model_args)

    # enable lora training
    if training_args.lora_enable:
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    # move to dtype
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              sign_data_args=sign_data_args,
                                              sign_multi_task_args=sign_multi_task_args,
                                              sign_multi_task_eval_args=sign_multi_task_eval_args)

    # print out the parameters that will be updated for training
    param_update = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name:
                param_update.add('lora adapters')
            else:
                param_update.add(name)
    print("[SignLLaVA]: The following parameters will be updated: ", sorted(param_update))

    # add encoder parameters for trainer
    training_args.encoder_params = sign_model_args

    tokenizer.save_pretrained(output_dir)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    # save the configuration.yaml
    if not training_args.resume_from_checkpoint:
        shutil.copy(extra_args.yaml_args, os.path.join(output_dir, "config.yaml"))
        model.config.save_pretrained(output_dir)

    # save the language prompt information
    language_prompt = {"system": conversation_lib.default_conversation.system,
                       "prompt": PROMPT_OPTIONS}
    with open(os.path.join(output_dir, "prompt.json"), "w") as prompt_json_file:
        json.dump(language_prompt, prompt_json_file)
    if training_args.resume_from_checkpoint and list(pathlib.Path(output_dir).glob("checkpoint-*")):
        print("[SignLLaVA]: resume training from checkpoint: ", output_dir)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
