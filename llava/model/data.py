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
from transformers import EarlyStoppingCallback
import tokenizers

from llava.constants import *
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_video_token

global PROMPT

# set prompt
    if sign_data_args["context_window_size"] + sign_data_args["prelude_window_size"] == 0:
        PROMPT = PROMPT_NO_CONTEXT
    else:
        PROMPT = PROMPT_CONTEXT

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
                 split: str):
        super(SignContextDataset, self).__init__()
        self.sign_data_args = sign_data_args
        self.tokenizer = tokenizer
        self.split = split
        data_dir = sign_data_args['data_dir']

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
        trans_dict = self.annotation[video_id][self.clip_order_from_int[video_id][clip_id]]
        translation = random.choice(trans_dict['paraphrases'] + [trans_dict['translation']])
    
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
        # get the visual features
        visual_features = {}
        for input_type in INPUT_TYPES:
            if eval(f"self.{input_type}") is not None:
                shard = self.h5shard[self.split][input_type][video_id]
                vf = torch.tensor(numpy.array(eval(f"self.{input_type}")[shard][video_id][clip_name]))
                visual_features[input_type] = vf
        src = {}
        src['id'] = "({0},{1})".format(str(video_id), str(clip_id))
        video_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN
        src['conversations'] = [{'from': 'human', 
                'value':video_token+'\n'+PROMPT.replace('<context>', ' '.join(context))},
                {'from': 'gpt',
                'value':translation}]

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
                    clip_id = self.clip_order_to_int[vi][ci]
                    h5_video_clip.add((vi, clip_id))
        return h5_video_clip
    

    def remove_missing_annotation(self, h5_video_clip):
        annotations_to_delete = set(self.list_data) - h5_video_clip
        for a in annotations_to_delete:
            self.list_data.remove(a)
 

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
                                sign_data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SignContextDataset(
        tokenizer = tokenizer,
        sign_data_args = sign_data_args,
        split = "train"
    )
    dev_dataset = SignContextDataset(
        tokenizer = tokenizer,
        sign_data_args = sign_data_args,
        split = "dev"
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                data_collator=data_collator)
