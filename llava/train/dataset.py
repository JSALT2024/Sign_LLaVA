import json
import os
import random
from typing import Dict

import cv2
import torch
import transformers
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.constants import *
from llava.mm_utils import tokenizer_video_token

local_rank = None


def load_video_cv(path: str) -> (list, int):
    """Returns list of frames in bgr"""
    video = []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
    cap.release()
    return video, fps


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


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
        assert role == conv.roles[j % 2], f"convo roles {j}"
        conv.append_message(role, sentence["value"])
    conversations.append(conv.get_prompt())
    # Tokenize conversations
    input_ids = torch.stack([tokenizer_video_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations],
                            dim=0)
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
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # the length of non-target (non-labels)
        # cur_len: the length of non-target parts
        parts = conversation.split(assistant_header)
        cur_len += len(tokenizer_video_token(parts[0], tokenizer))
        target[:cur_len] = IGNORE_INDEX
        for part in parts[1:]:
            if part != "":
                target[cur_len:cur_len + assistant_header_len] = IGNORE_INDEX
                cur_len += assistant_header_len
                response_eot_id = part.find(eot)
                response_len = len(tokenizer_video_token(part[:response_eot_id], tokenizer)) + 1
                cur_len += response_len
                if cur_len < total_len:
                    part_res = part[response_eot_id + len(eot) + 1:]
                    part_res_len = len(tokenizer_video_token(part_res, tokenizer))
                    target[cur_len:cur_len + part_res_len] = IGNORE_INDEX
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
                 sign_multi_task_eval_args: dict = None,
                 split: str = 'train'):
        super(SignContextDataset, self).__init__()
        self.sign_data_args = sign_data_args
        self.sign_multi_task_args = sign_multi_task_args
        self.sign_multi_task_eval_args = sign_multi_task_eval_args
        self.tokenizer = tokenizer
        self.split = split
        self.keep_prediction_keys = ['cropped_keypoints', 'bbox_left_hand', 'bbox_right_hand', 'bbox_face']

        # prepare task "args"
        self.tasks = self.get_tasks()  # {"translation": 0.4, "one_word_present": 0.2, "multi_word_present": 0.2, "is_reversed": 0.2}
        if self.sign_multi_task_eval_args is not None:
            self.eval_tasks = self.get_tasks_eval()

        # load annotations
        data_dir = sign_data_args['data_dir']
        annotation_path = sign_data_args['annotation_path'][self.split]
        annotation_path = os.path.join(data_dir, annotation_path)
        self.annotation = json.load(open(annotation_path, "r"))

        # build keyword vocabulary
        self.keyword_vocabulary = {}
        for video_id in self.annotation:
            for clip_name in self.annotation[video_id]:
                if clip_name != "clip_order":
                    clip_dict = self.annotation[video_id][clip_name]
                    keywords = clip_dict['keywords']
                    if keywords != []:
                        self.keyword_vocabulary.update({k: 1 for k in keywords})

        # load clip order
        self.clip_order_to_int = {}
        self.clip_order_from_int = {}
        for video_id in self.annotation.keys():
            co = self.annotation[video_id]['clip_order']
            self.clip_order_from_int[video_id] = dict(zip(range(len(co)), co))
            self.clip_order_to_int[video_id] = dict(zip(co, range(len(co))))

        self.list_data = []
        for video_id, clip_dict in self.annotation.items():
            for clip_name in clip_dict:
                if clip_name != "clip_order":
                    self.list_data.append((video_id, self.clip_order_to_int[video_id][clip_name]))

        # TODO: remove missing

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        video_id, clip_id = self.list_data[i]
        clip_name = self.clip_order_from_int[video_id][clip_id]

        # Get context: concatenate preceding sentences,
        # the number of sentences is defined by data_args.context_window_size
        context = []
        total_num_preceding_sents = clip_id
        context_window_size = self.sign_data_args['context_window_size']
        prelude_window_size = self.sign_data_args['prelude_window_size']
        if total_num_preceding_sents >= context_window_size + prelude_window_size:
            preceding_ids = list(range(prelude_window_size)) + \
                            list(range(clip_id - context_window_size, clip_id))
        else:
            preceding_ids = range(total_num_preceding_sents)
        for ci in preceding_ids:
            preceding_clip_name = self.clip_order_from_int[video_id][ci]
            context.append(self.annotation[video_id][preceding_clip_name]['translation'])

        src = {}
        src['id'] = f"({str(video_id)},{str(clip_id)})"
        video_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN

        sampled_task, text_prompt, response = self.get_task_prompt(video_id, clip_name, context)

        # get the visual features
        visual_features = {}
        clip_path = os.path.join(self.sign_data_args['clip_dir'], f"{clip_name}.mp4")
        json_path = os.path.join(self.sign_data_args['clip_dir'], f"{clip_name}.json")

        clip_data, fps = load_video_cv(clip_path)
        json_data = load_json(json_path)

        # TODO: normalize keypoints for pose encoder
        json_data = self.process_prediction_data(json_data, self.keep_prediction_keys)

        # augment data
        if sampled_task == "is_reversed" and response == "yes":
            clip_data.reverse()
            [v.reverse() for v in json_data.values()]

        visual_features["clip"] = clip_data
        visual_features["predictions"] = json_data

        src['conversations'] = [
            {'from': 'human', 'value': video_token + '\n' + text_prompt},
            {'from': 'gpt', 'value': response}
        ]

        # <video> -> <video_start><video><video_end>
        data_dict = preprocess_llama_3(src, self.tokenizer)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])
        data_dict['visual_features'] = visual_features
        video_sep = DEFAULT_VIDEO_END_TOKEN + DEFAULT_VIDEO_START_TOKEN
        data_dict['video_sep_ids'] = tokenizer_video_token(video_sep, self.tokenizer, return_tensors='pt')
        return data_dict

    def process_prediction_data(self, prediction: dict, keep_keys: list) -> dict:
        _predictions = {}
        for key in prediction.keys():
            if key not in keep_keys:
                continue
            _predictions[key] = prediction[key]
        return _predictions

    def remove_missing_annotation(self, h5_video_clip):
        annotations_to_delete = set(self.list_data) - h5_video_clip
        for a in annotations_to_delete:
            self.list_data.remove(a)

    def get_tasks(self):
        tasks = {}
        for task in self.sign_multi_task_args:
            if self.sign_multi_task_args[task]['sample_weight'] > 0:
                tasks[task] = self.sign_multi_task_args[task]['sample_weight']
        return tasks

    def get_tasks_eval(self):
        tasks = {}
        for task in self.sign_multi_task_eval_args:
            if self.sign_multi_task_eval_args[task]['sample_weight'] > 0:
                tasks[task] = self.sign_multi_task_eval_args[task]['sample_weight']
        return tasks

    def get_task_prompt(self, video_id, clip_name, context):
        # {"translation": 0.4, "one_word_present": 0.2, "multi_word_present": 0.2, "is_reversed": 0.2}
        clip_dict = self.annotation[video_id][clip_name]
        if self.split == 'train' or (self.split == 'dev' and self.sign_multi_task_eval_args is not None):
            if self.split == 'train':
                sampled_task = random.choices(list(self.tasks.keys()), weights=list(self.tasks.values()), k=1)[0]
            else:
                sampled_task = \
                    random.choices(list(self.eval_tasks.keys()), weights=list(self.eval_tasks.values()), k=1)[0]
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
                # num_positive_keywords = random.randint(0, min(len(keywords), num_keywords))
                num_positive_keywords = min(len(keywords), int(num_keywords / 2))
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
                # shuffled_responses, shuffled_keywords = zip(*random.sample(list(zip(responses, chosen_keywords)), len(responses)))
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
