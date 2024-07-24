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

import os
import copy
import random
import shutil
from dataclasses import dataclass, field
import json
import h5py
import yaml
from collections import defaultdict
import logging
import pathlib
import numpy
from typing import Dict, Optional, Sequence, List

import torch

import transformers
from transformers import set_seed
import tokenizers

from llava.constants import *
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_video_token

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta/Meta-Llama-3-8B-Instruct")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=".")
    bf16: bool = field(default=True)
    report_to: Optional[str] = field(default="wandb")
    #gradient_accumulation_steps: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="steps")
    metric_for_best_model: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    resume_from_checkpoint: bool = field(default=False)
    run_name: Optional[str] = field(default=None)
    label_smoothing_factor: Optional[float] = field(default=0.1)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

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

    if 'lm_head' in lora_module_names: # needed for 16-bit
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

def preprocess_llama_3(
    source,
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    conv.messages = []
    for j, sentence in enumerate(source['conversations']):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])
    conversations.append(conv.get_prompt())
    # Tokenize conversations
    input_ids = torch.stack([tokenizer_video_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    # targets: masked input_ids, where only the assitant inputs are kept, 
    #          and all the previous tokens are masked with IGNORE_INDEX -100
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    bot = "<|begin_of_text|>"
    eot = "<|eot_id|>" 
    
    assistant_header_len = len(tokenizer_video_token(assistant_header, tokenizer))
    for conversation, target in zip(conversations, targets):
        cur_len = 0
        # targets: labels of assistant output
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # the length of non-target (non-labels)
        # cur_len: the length of non-target parts
        parts = conversation.split(assistant_header)
        cur_len += len(tokenizer_video_token(parts[0], tokenizer))
        target[:cur_len] = IGNORE_INDEX
        for part in parts[1:]:
            if part != "":
                target[cur_len:cur_len+assistant_header_len] = IGNORE_INDEX
                cur_len += assistant_header_len
                response_eot_id = part.find(eot)
                response_len = len(tokenizer_video_token(part[:response_eot_id], tokenizer)) + 1
                cur_len += response_len
                if cur_len < total_len:
                    part_res = part[response_eot_id+len(eot)+1:]
                    part_res_len = len(tokenizer_video_token(part_res, tokenizer))
                    target[cur_len:cur_len+part_res_len] = IGNORE_INDEX
                    cur_len += part_res_len
        if cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
            )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class SignContextDataset(Dataset):
    """Dataset for supervised training for sign language translation with context."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 sign_data_args: dict,
                 sign_multi_task_args: dict,
                 split: str):
        super(SignContextDataset, self).__init__()
        self.sign_data_args = sign_data_args
        self.sign_multi_task_args = sign_multi_task_args
        self.tokenizer = tokenizer
        self.split = split
        data_dir = sign_data_args['data_dir']
        self.tasks = self.get_tasks() # {"translation": 0.4, "one_word_present": 0.2, "multi_word_present": 0.2, "is_reversed": 0.2}

        if self.split == "train":
            annotation_path = sign_data_args['annotation_path']['train']
        elif self.split == "dev":
            annotation_path = sign_data_args['annotation_path']['dev']
        elif self.split == "test":
            annotation_path = sign_data_args['annotation_path']['test']
        annotation_path = os.path.join(data_dir, annotation_path)
        self.annotation = json.load(open(annotation_path, "r"))
        # {{video_id: {clip_id: {"translation": ..., "paraphrases": [A, B, C] }}},
        # }
        # self.list_data: i -> (video_id, clip_id)
        self.list_data = [] # [(video_id, clip_id), ...]
        
        # build keyword vocabulary if needed
        if self.split == "train" and ("one_word_present" in self.tasks or "multi_word_present" in self.tasks):
            self.keyword_vocabulary = {}
            for video_id in self.annotation:
                for clip_name in self.annotation[video_id]:
                    if clip_name != "clip_order":
                        clip_dict = self.annotation[video_id][clip_name]
                        keywords = clip_dict['keywords']
                        if keywords != []:
                            self.keyword_vocabulary.update({k: 1 for k in keywords})

        self.h5shard = defaultdict(lambda: defaultdict(dict))
        self.clip_order_to_int = {}
        self.clip_order_from_int = {}
        for video_id in self.annotation.keys():
            co = self.annotation[video_id]['clip_order']
            self.clip_order_from_int[video_id] =  dict(zip(range(len(co)),co))
            self.clip_order_to_int[video_id] =  dict(zip(co,range(len(co))))

        for video_id, clip_dict in self.annotation.items():
            for clip_name in clip_dict:
                if clip_name != "clip_order":
                    self.list_data.append((video_id, self.clip_order_to_int[video_id][clip_name]))
        for input_type in INPUT_TYPES:
            enable_feature = sign_data_args['visual_features'][input_type]['enable_input']
            if 'train' in sign_data_args['visual_features'][input_type]:
                vf_train_path = sign_data_args['visual_features'][input_type]['train']
                vf_train_path = os.path.join(data_dir, vf_train_path)
            else:
                vf_train_path = None
            if 'dev' in sign_data_args['visual_features'][input_type]:
                vf_dev_path = sign_data_args['visual_features'][input_type]['dev']
                vf_dev_path = os.path.join(data_dir, vf_dev_path)
            else:
                vf_dev_path = None
            if 'test' in sign_data_args['visual_features'][input_type]:
                vf_test_path = sign_data_args['visual_features'][input_type]['test']
                vf_test_path = os.path.join(data_dir, vf_test_path)
            else:
                vf_test_path = None
            if enable_feature and vf_train_path is not None and vf_dev_path is not None:
                if self.split == "train":
                    h5_video_clip = self.read_multih5_json(data_dir, vf_train_path, input_type)
                    self.remove_missing_annotation(h5_video_clip)
                elif self.split == "dev":
                    h5_video_clip = self.read_multih5_json(data_dir, vf_dev_path, input_type)
                    self.remove_missing_annotation(h5_video_clip)
            elif self.split == "test" and enable_feature and vf_test_path is not None:
                h5_video_clip = self.read_multih5_json(data_dir, vf_test_path, input_type)
                self.remove_missing_annotation(h5_video_clip)
            else:
                exec(f"self.{input_type}=None")
        # self.sign2vec_train: {video_id: {clip_id: R(NxV), clip_id:...}, ..., ...}, 
        #   {video_id: ........}}}
        # self.sign2vec_dev: ...
    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        video_id, clip_id = self.list_data[i]
        # sources: json, 'image': 'train/06December_2011_Tuesday_tagesschau-6843'
        clip_name = self.clip_order_from_int[video_id][clip_id]
        # Get context: concatenate preceding sentences, 
        # the number of sentences is defined by data_args.context_window_size
        context = []
        total_num_preceding_sents = clip_id
        context_window_size = self.sign_data_args['context_window_size']
        prelude_window_size = self.sign_data_args['prelude_window_size']
        if total_num_preceding_sents >=  context_window_size + prelude_window_size:
            preceding_ids = list(range(prelude_window_size)) + \
                            list(range(clip_id-context_window_size, clip_id))
        else:
            preceding_ids = range(total_num_preceding_sents)
        for ci in preceding_ids:    
            preceding_clip_name = self.clip_order_from_int[video_id][ci]
            context.append(self.annotation[video_id][preceding_clip_name]['translation'])
        
        src = {}
        src['id'] = "({0},{1})".format(str(video_id), str(clip_id))
        video_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN

        sampled_task, text_prompt, response = self.get_task_prompt(video_id, clip_name, context)

        # get the visual features
        visual_features = {}
        for input_type in INPUT_TYPES:
            if eval(f"self.{input_type}") is not None:
                shard = self.h5shard[self.split][input_type][video_id]
                vf = torch.tensor(numpy.array(eval(f"self.{input_type}")[shard][video_id][clip_name]))
                # do augmentation
                if sampled_task == "is_reversed" and response == "yes":
                    vf = vf.flip(0)
                visual_features[input_type] = vf
            
        src['conversations'] = [{'from': 'human', 
                'value':video_token+'\n'+text_prompt},
                {'from': 'gpt',
                'value':response}]

        # <video> -> <video_start><video><video_end>
        data_dict = preprocess_llama_3(src, self.tokenizer)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])
        data_dict['visual_features'] = visual_features
        video_sep = DEFAULT_VIDEO_END_TOKEN+DEFAULT_VIDEO_START_TOKEN
        data_dict['video_sep_ids'] = tokenizer_video_token(video_sep, self.tokenizer, return_tensors='pt')
        return data_dict

    def read_multih5_json(self, data_dir, json_filename, input_type):
        """Helper function for reading json specifications of multiple H5 files for visual features"""
        h5_video_clip = set()
        with open(os.path.join(data_dir, json_filename), 'r') as F:
            self.h5shard[self.split][input_type] = json.load(F)
            exec(f"self.{input_type} = dict()")
            print(f"{input_type}: {self.split} data is loaded from: ")
            for k in set(self.h5shard[self.split][input_type].values()):
                h5file = os.path.join(data_dir, json_filename.replace('metadata_','').replace('.json',".%s.h5"%k))
                print("--" + h5file) #,k,json_filename,data_dir)
                exec(f"self.{input_type}[k] = h5py.File(h5file, 'r')")

            for vi in eval(f"self.{input_type}[k]").keys():
                for ci in eval(f"self.{input_type}[k][vi]").keys():
                    if vi in self.clip_order_to_int:
                        if ci in self.clip_order_to_int[vi]:
                            clip_id = self.clip_order_to_int[vi][ci]
                            h5_video_clip.add((vi, clip_id))
        return h5_video_clip
    

    def remove_missing_annotation(self, h5_video_clip):
        annotations_to_delete = set(self.list_data) - h5_video_clip
        for a in annotations_to_delete:
            self.list_data.remove(a)
    
    def get_tasks(self):
        tasks = {}
        for task in self.sign_multi_task_args:
            if self.sign_multi_task_args[task]['enable'] and self.sign_multi_task_args[task]['sample_weight'] > 0:
                tasks[task] = self.sign_multi_task_args[task]['sample_weight']
        return tasks
    
    def get_task_prompt(self, video_id, clip_name, context):
        # {"translation": 0.4, "one_word_present": 0.2, "multi_word_present": 0.2, "is_reversed": 0.2}
        clip_dict = self.annotation[video_id][clip_name]
        if self.split == 'train':
            sampled_task = random.choices(list(self.tasks.keys()), weights=list(self.tasks.values()), k=1)[0]
            if sampled_task == "translation": 
                if self.sign_data_args.get('use_paraphrases', False):
                    translation = random.choice(clip_dict['paraphrases'] + [clip_dict['translation']])
                else:
                    translation = clip_dict['translation']
                if context == []:
                    text_prompt = PROMPT_OPTIONS["translate_no_context"]
                else:
                    text_prompt = PROMPT_OPTIONS["translate_with_context"].replace('<context>', ' '.join(context))
                response = translation
            elif sampled_task == "one_word_present":
                translation = clip_dict['translation']
                negative_keyword = random.choice(list(self.keyword_vocabulary.keys()))
                while negative_keyword in translation:
                    negative_keyword = random.choice(list(self.keyword_vocabulary.keys()))
                keywords = clip_dict['keywords']
                if keywords != []:
                    positive_keyword = random.choice(keywords)
                    chosen_keyword = random.choice([positive_keyword, negative_keyword])
                else:
                    chosen_keyword = negative_keyword
                chosen_keyword = chosen_keyword.lower()
                text_prompt = PROMPT_OPTIONS["one_word_present"].replace('<word>', chosen_keyword)
                response = "yes" if chosen_keyword in clip_dict['keywords'] else "no"
            elif sampled_task == "multi_word_present":
                translation = clip_dict['translation']
                keywords = clip_dict['keywords']
                num_keywords = random.randint(2, self.sign_multi_task_args['multi_word_present']['max_num_words'])
                chosen_keywords = []
                responses = []
                #num_positive_keywords = random.randint(0, min(len(keywords), num_keywords))
                num_positive_keywords = min(len(keywords), int(num_keywords/2))
                num_negative_keywords = num_keywords - num_positive_keywords
                if num_positive_keywords > 0:
                    positive_keywords = random.sample(keywords, num_positive_keywords)
                    for keyword in positive_keywords:
                        chosen_keywords.append(keyword)
                        responses.append("yes")
                if num_negative_keywords > 0:
                    for _ in range(num_negative_keywords):
                        negative_keyword = random.choice(list(self.keyword_vocabulary.keys()))
                        while negative_keyword in translation or negative_keyword in chosen_keywords:
                            negative_keyword = random.choice(list(self.keyword_vocabulary.keys()))
                        chosen_keywords.append(negative_keyword)
                        responses.append("no")
                #shuffled_responses, shuffled_keywords = zip(*random.sample(list(zip(responses, chosen_keywords)), len(responses)))
                # order keywords by alphabetical order
                chosen_keywords = [w.lower() for w in chosen_keywords]
                sorted_pairs = sorted(zip(responses, chosen_keywords), key=lambda x: x[1])
                shuffled_responses, shuffled_keywords = zip(*sorted_pairs)
                text_prompt = PROMPT_OPTIONS["multi_words_present"].replace('<words>', ', '.join(shuffled_keywords))
                response = ', '.join(shuffled_responses)
            elif sampled_task == "is_reversed":
                text_prompt = PROMPT_OPTIONS["is_reversed"]
                response = "yes" if random.random() < 0.5 else "no"
        else:
            response = clip_dict['translation']
            if context == []:
                text_prompt = PROMPT_OPTIONS["translate_no_context"]
            else:
                text_prompt = PROMPT_OPTIONS["translate_with_context"].replace('<context>', ' '.join(context))
            sampled_task = "translation"
        return sampled_task, text_prompt, response

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, video_sep_ids, visual_features = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "video_sep_ids", "visual_features"))
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
                                sign_multi_task_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SignContextDataset(
        tokenizer = tokenizer,
        sign_data_args = sign_data_args,
        sign_multi_task_args = sign_multi_task_args,
        split = "train"
    )
    dev_dataset = SignContextDataset(
        tokenizer = tokenizer,
        sign_data_args = sign_data_args,
        sign_multi_task_args = sign_multi_task_args,
        split = "dev"
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
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, ExtraArguments))
    model_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    with open(extra_args.yaml_args, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
        update_arguments(model_args, yaml_config['ModelArguments'])
        update_arguments(training_args, yaml_config['TrainingArguments'])
    sign_data_args = yaml_config["SignDataArguments"]
    sign_model_args = yaml_config["SignModelArguments"]
    sign_multi_task_args = yaml_config["SignMultiTaskArguments"]

    # set seed
    set_same_seed(training_args.seed)
    
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    projector_lst = []
    for input_type in sign_data_args['visual_features']:
        if eval(f"sign_data_args['visual_features']['{input_type}']['enable_input']"):
            projector_lst.append(f"{input_type}_projector")
    skip_modules = projector_lst + ['lm_head']

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            #device_map="auto",
            device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int4_skip_modules=skip_modules,
                llm_int8_skip_modules=skip_modules,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False, # must be false if --bf True
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    model = SignLlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                local_files_only=True,
                sign_model_args=sign_model_args,
                sign_data_args=sign_data_args,
                **bnb_model_from_pretrained_args
            )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
        tokenizer.add_special_tokens({"unk_token":"<unk>"})
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
            
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for input_type in INPUT_TYPES:
            projector = eval(f"model.get_model().{input_type}_projector")
            if projector is not None:
                for p in projector.parameters():
                    p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        projector = eval(f"model.get_model().{input_type}_projector")
        if projector is not None:
            for p in projector.parameters():
                p.requires_grad = False

    if training_args.bits in [4, 8]:
        for input_type in INPUT_TYPES:
            projector = eval(f"model.get_model().{input_type}_projector")
            if projector is not None:
                projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

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
                                              sign_multi_task_args=sign_multi_task_args)
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    # save the configuration.yaml
    os.makedirs(training_args.output_dir, exist_ok=True)
    shutil.copy(extra_args.yaml_args, os.path.join(training_args.output_dir, "config.yaml"))
    model.config.save_pretrained(training_args.output_dir)

    # save the language prompt information
    language_prompt = {"system": conversation_lib.default_conversation.system,
                       "prompt": PROMPT_OPTIONS}
    with open(os.path.join(training_args.output_dir, "prompt.json"), "w") as prompt_json_file:
        json.dump(language_prompt, prompt_json_file)
    if training_args.resume_from_checkpoint and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

if __name__ == "__main__":
    train()
