import argparse
import torch
import json
import pickle
from tqdm import tqdm
import os
import json
import h5py
import yaml
import random
import numpy
from collections import defaultdict
import tqdm
import shutil

import transformers
from transformers import set_seed
import tokenizers
from llava.constants import *
from torch.utils.data import Dataset
from llava.model.signbuilder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_video_token
from llava import conversation as conversation_lib

import requests
from io import BytesIO
import re

global generation
generation = defaultdict(dict)

def preprocess_llama_3(
    source,
    tokenizer: transformers.PreTrainedTokenizer
):
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
                split: str,
                prompt: str,
                dtype):
        super(SignContextDataset, self).__init__()
        self.sign_data_args = sign_data_args
        self.tokenizer = tokenizer
        self.split = split
        self.prompt = prompt
        self.dtype = dtype
        data_dir = sign_data_args['data_dir']

        annotation_path = sign_data_args['annotation_path']
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

        for video_id in self.annotation:
            for clip_name in self.annotation[video_id]['clip_order']:
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

    def __getitem__(self, i):
        video_id, clip_id = self.list_data[i]
        # sources: json, 'image': 'train/06December_2011_Tuesday_tagesschau-6843'
        clip_name = self.clip_order_from_int[video_id][clip_id]
        trans_dict = self.annotation[video_id][self.clip_order_from_int[video_id][clip_id]]
        translation = [trans_dict['translation']]
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
            if self.sign_data_args.get("prepared_predicted_context", False):
                context.append(self.annotation[video_id][preceding_clip_name]['hypothesis'])
            elif self.sign_data_args.get("on_the_fly_predicted_context", False):
                context.append(generation[video_id][preceding_clip_name]['hypothesis'])
            else:
                context.append(self.annotation[video_id][preceding_clip_name]['translation'])
        # get the visual features
        visual_features = {}
        for input_type in INPUT_TYPES:
            if eval(f"self.{input_type}") is not None:
                shard = self.h5shard[self.split][input_type][video_id]
                vf = torch.tensor(numpy.array(eval(f"self.{input_type}")[shard][video_id][clip_name])).to(self.dtype)
                visual_features[input_type] = vf
        src = {}
        src['id'] = "({0},{1})".format(str(video_id), str(clip_id))
        video_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN
        
        src['conversations'] = [{'from': 'human', 
                'value':video_token+'\n'+self.prompt.replace('<context>', ' '.join(context))},
                {'from': 'gpt',
                'value':translation}]

        # <video> -> <video_start><video><video_end>
        data_dict = preprocess_llama_3(src, self.tokenizer)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])
        data_dict['visual_features'] = visual_features
        video_sep = DEFAULT_VIDEO_END_TOKEN+DEFAULT_VIDEO_START_TOKEN
        data_dict['video_sep_ids'] = tokenizer_video_token(video_sep, self.tokenizer, return_tensors='pt')
        data_dict['video_id'] = video_id
        data_dict['clip_name'] = clip_name
        data_dict['reference'] = translation
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

def set_same_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

def eval_model(config_yaml):
    with open(config_yaml, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    checkpoint_num = config['GenerateArguments']['checkpoint_num'] 
    shutil.copy(config_yaml, os.path.join(config['GenerateArguments']['model_path'], f"generation-{checkpoint_num}.yaml"))

    set_same_seed(config['GenerateArguments']['seed'])

    dtype = torch.bfloat16 if config['GenerateArguments']['bf16'] else torch.float16
    # Model
    disable_torch_init()

    tokenizer, model, model_max_length = load_pretrained_model(config, use_flash_attn=False)
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token":"<unk>"})
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    video_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN

    # set prompt
    sign_data_args = config['SignDataArguments']
    if sign_data_args["context_window_size"] + sign_data_args["prelude_window_size"] == 0:
        prompt = PROMPT_NO_CONTEXT
    else:
        prompt = PROMPT_CONTEXT

    test_dataset = SignContextDataset(
        tokenizer = tokenizer,
        sign_data_args = sign_data_args,
        split = "test",
        prompt = prompt,
        dtype=dtype
    )

    generate_kwargs = {
        "max_new_tokens": config['GenerateArguments']['max_new_tokens'],
        "temperature": config['GenerateArguments']['temperature'],
        "top_p": config['GenerateArguments']['top_p'],
        "num_beams": config['GenerateArguments']['num_beams'],
        "min_length": config['GenerateArguments']['min_length'],
        "do_sample": config['GenerateArguments']['do_sample'],
        "remove_invalid_values": config['GenerateArguments']['remove_invalid_values'],
        }
    '''
    - greedy decoding if num_beams=1 and do_sample=False
    - contrastive search if penalty_alpha>0 and top_k>1
    - multinomial sampling if num_beams=1 and do_sample=True
    - beam-search decoding if num_beams>1 and do_sample=False
    - beam-search multinomial sampling if num_beams>1 and do_sample=True
    - diverse beam-search decoding if num_beams>1 and num_beam_groups>1
    - constrained beam-search decoding if constraints!=None or force_words_ids!=None
    - assisted decoding if assistant_model or prompt_lookup_num_tokens is passed to .generate()
    '''

    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    data_dir = sign_data_args['data_dir']
    annotation_path = sign_data_args['annotation_path']
    annotation_path = os.path.join(data_dir, annotation_path)
    annotation = json.load(open(annotation_path, "r"))
    
    if config['GenerateArguments']['num_datapoints'] == -1:
        num_datapoints = len(test_dataset)
    else:
        num_datapoints = min(config['GenerateArguments']['num_datapoints'], len(test_dataset))
    with torch.inference_mode():
        for i in tqdm.tqdm(range(num_datapoints)):
            data_dict = test_dataset[i]
            input_ids = data_dict['input_ids']
            labels = data_dict['labels']
            translation = data_dict['reference']
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids.unsqueeze(0),
                batch_first=True,
                padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels.unsqueeze(0),
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :model_max_length].to(model.device)
            labels = labels[:, :model_max_length].to(model.device)
            output_dict = model.generate(
                inputs = input_ids,
                labels = labels,
                visual_features = [data_dict['visual_features']],
                video_sep_ids = [data_dict['video_sep_ids']],
                pad_token_id = tokenizer.unk_token_id,
                output_scores = True,
                return_dict_in_generate=True,
                forced_bos_token_id = tokenizer.encode("\n\n")[1:],
                decoder_start_token_id = tokenizer.encode("\n\n")[1:],
                bos_token_id = tokenizer.encode("\n\n")[1],
                **generate_kwargs
            )
            scores = torch.stack(list(output_dict['scores'])).to(dtype).unsqueeze(0).squeeze(2)
            output_ids = output_dict['sequences']
            outputs = tokenizer.batch_decode(output_ids, 
                skip_special_tokens=config['GenerateArguments']['skip_special_tokens'])[0].strip()
            if config['GenerateArguments']['do_verbose']:
                print("reference:", translation[0])
                print("outputs:", outputs)
                print("scores", scores)
                print("output_ids", output_ids)
            generation[data_dict['video_id']]['clip_order'] = annotation[data_dict['video_id']]['clip_order']
            generation[data_dict['video_id']][data_dict['clip_name']] = annotation[data_dict['video_id']][data_dict['clip_name']]
            generation[data_dict['video_id']][data_dict['clip_name']]['hypothesis'] = outputs
            
    with open(os.path.join(config['GenerateArguments']['model_path'], config['GenerateArguments']['generate_des']), 'w') as gen_file:
        json.dump(generation, gen_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_args", type=str, default="signllava/configs/generate.yaml")
    args = parser.parse_args()

    eval_model(args.yaml_args)