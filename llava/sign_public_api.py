# This file is the public API of Sign_LLaVA that is used by the demo
# application. Anotherwords, this is the *only* place, where the demo
# and this repository meet.
#
# It exposes the `SignLlava` class, which represents the loaded sign llava model
# and this class exposes the `run_inference` method, which is the only method
# called by the demo. The methods gets a `SignLlavaInput` instance and returns
# a `SignLlavaOutput` instance. The `run_inference` method translates a single
# clip, not the entire video. Therefore the previously translated context text
# must be passed in by the caller.


import os
import numpy as np
import random
from typing import Optional, Any
from collections import defaultdict
import dataclasses
import yaml
import torch

import transformers
from transformers import set_seed
import tokenizers

from llava.model.signbuilder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_video_token
from llava import conversation as conversation_lib
from llava.constants import PROMPT_OPTIONS, DEFAULT_VIDEO_END_TOKEN, \
    DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, \
    INPUT_TYPES


SYSTEM_PROMPT=\
    "You are a specialized assistant for American Sign Language (ASL). "\
    "Your capabilities include understanding and interpreting ASL content "\
    "from videos provided by users. You can translate ASL into English "\
    "accurately and respond to specific inquiries about the presence of "\
    "particular signs within the video, demonstrating comprehension. "\
    "Your role is to assist users effectively by leveraging sign language "\
    "for communication and tasks. The video is represented by a single or "\
    "several streams of visual features delimited by <video_start> and <video_end>."


def prepare_translation_prompt(context: Optional[str]) -> str:
    if context is None:
        return PROMPT_OPTIONS["translate_no_context"]
    return PROMPT_OPTIONS["translate_with_context"].replace("<context>", context)


@dataclasses.dataclass
class GenerationConfig:
    """Contains generation hyperparameters, such as temperature"""
    max_new_tokens: int = 200
    temperature: float = 0.1
    top_p: int = 1
    num_beams: int = 5
    min_length: int = 1
    do_sample: bool = False
    remove_invalid_values: bool = True
    #skip_special_tokens: bool = True


@dataclasses.dataclass
class SignLlavaInput:
    """Container for all the Sign LLaVA inference input data.

    The input data represents all the data necessary
    to translate one clip (not the whole video!).

    The primary purpose of this class is to group together all the input
    data of the LLM model so that it can be run as a black box. The secondary
    purpose is to validate the structure of this data, making sure no unexpected
    garbage leaks through. This class should NOT do any data processing.
    """

    sign2vec_features: Optional[np.ndarray]
    """
    Contains a numpy array of shape [TIME, FEATURES] of dtype float32 which
    contains the features predicted by the Sign2Vec model. The value can be
    None, because the s2v encoder can be disbled in the demo application.
    The temporal dimension is smaller than for the other features, because
    sign2vec does temporal compression (at least 2x smaller).
    """

    mae_features: Optional[np.ndarray]
    """
    Contains a numpy array of shape [TIME, FEATURES] of dtype float32 which
    contains the features predicted by the MAE model. The value can be
    None, because the MAE encoder can be disbled in the demo application.
    The temporal dimension equals the number of frames in the video clip.
    """

    dino_features: Optional[np.ndarray]
    """
    Contains a numpy array of shape [TIME, FEATURES] of dtype float32 which
    contains the features predicted by the DINO model. The value can be
    None, because the DINO encoder can be disbled in the demo application.
    The temporal dimension equals the number of frames in the video clip.
    """

    prompt: str
    """
    Contains the user prompt to the LLM depending on the task to be performed
    (e.g. translation, keywords, etc.). The prompt is a string text.
    To obtain the prompts that the model was trained on, see the PROMPT_OPTIONS
    constant available in this file. If the prompt expects a text value to place
    within it (e.g. the translation context), it's the caller's responsibility
    to provide this value before passing the final prompt here.

    If chatting-history or system prompt is to be added, this field should get
    a dedicated class (e.g. PromptData).
    """
    generation_config: GenerationConfig

    def __post_init__(self):
        if self.sign2vec_features is not None:
            assert str(self.sign2vec_features.dtype) == "float32"
            assert len(self.sign2vec_features.shape) == 2
            assert self.sign2vec_features.shape[0] > 0
            assert self.sign2vec_features.shape[1] == 768
        if self.mae_features is not None:
            assert str(self.mae_features.dtype) == "float32"
            assert len(self.mae_features.shape) == 2
            assert self.mae_features.shape[0] > 0
            assert self.mae_features.shape[1] == 768
        if self.dino_features is not None:
            assert str(self.dino_features.dtype) == "float32"
            assert len(self.dino_features.shape) == 2
            assert self.dino_features.shape[0] > 0
            assert self.dino_features.shape[1] == 1152
        assert type(self.prompt) is str


@dataclasses.dataclass
class SignLlavaOutput:
    """
    Container for all the Sign LLaVA inference output data,
    including data used for visualization of the internals of the model
    """

    output: str
    """
    Contains the text output of the model. This is the translated text,
    or the yes/no keyword answer or any other thing the model generates.
    This depends on the user prompt.
    """

    sign2vec_embeddings: Optional[np.ndarray]
    """
    Sign2Vec features after being passed through the projector.
    It has shape [TIME, FEATURES].
    None if no s2v features were provided on the input.
    """

    mae_embeddings: Optional[np.ndarray]
    """
    MAE features after being passed through the projector.
    It has shape [TIME, FEATURES].
    None if no MAE features were provided on the input.
    """

    dino_embeddings: Optional[np.ndarray]
    """
    DINO features after being passed through the projector.
    It has shape [TIME, FEATURES].
    None if no DINO features were provided on the input.
    """

    def __post_init__(self):
        assert type(self.output) is str
        if self.sign2vec_embeddings is not None:
            assert str(self.sign2vec_embeddings.dtype) == "float32"
            assert len(self.sign2vec_embeddings.shape) == 2
            assert self.sign2vec_embeddings.shape[0] > 0
        if self.mae_embeddings is not None:
            assert str(self.mae_embeddings.dtype) == "float32"
            assert len(self.mae_embeddings.shape) == 2
            assert self.mae_embeddings.shape[0] > 0
        if self.dino_embeddings is not None:
            assert str(self.dino_embeddings.dtype) == "float32"
            assert len(self.dino_embeddings.shape) == 2
            assert self.dino_embeddings.shape[0] > 0
        # NOTE: the embedding dimensions depend on the llama model chosen,
        # they may be 4K or 8K


class SignLlava:
    """
    Represents the loaded Sign LLaVA model that can be used for inference.
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        model_max_length: int,
        config: dict
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_max_length = model_max_length
        self.config = config

    @staticmethod
    def load_from_checkpoint(checkpoint_folder_path: str) -> "SignLlava":
        config_path = os.path.join(checkpoint_folder_path, "api_config.yaml")

        # load the config yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # set model path in the config
        config["GenerateArguments"]["model_path"] = checkpoint_folder_path

        tokenizer, model, model_max_length = load_pretrained_model(config, use_flash_attn=False)
        if tokenizer.unk_token is None:
            tokenizer.add_special_tokens({"unk_token":"<unk>"})
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"

        return SignLlava(
            tokenizer=tokenizer,
            model=model,
            model_max_length=model_max_length,
            config=config
        )

    def preprocess_llama_3(self, source):
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
        input_ids = torch.stack([tokenizer_video_token(prompt, self.tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

        # Mask targets
        # targets: masked input_ids, where only the assitant inputs are kept, 
        #          and all the previous tokens are masked with IGNORE_INDEX -100
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        bot = "<|begin_of_text|>"
        eot = "<|eot_id|>" 
        
        assistant_header_len = len(tokenizer_video_token(assistant_header, self.tokenizer))
        for conversation, target in zip(conversations, targets):
            cur_len = 0
            # targets: labels of assistant output
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum()) # the length of non-target (non-labels)
            # cur_len: the length of non-target parts
            parts = conversation.split(assistant_header)
            cur_len += len(tokenizer_video_token(parts[0], self.tokenizer))
            target[:cur_len] = IGNORE_INDEX
            for part in parts[1:]:
                if part != "":
                    target[cur_len:cur_len+assistant_header_len] = IGNORE_INDEX
                    cur_len += assistant_header_len
                    response_eot_id = part.find(eot)
                    response_len = len(tokenizer_video_token(part[:response_eot_id], self.tokenizer)) + 1
                    cur_len += response_len
                    if cur_len < total_len:
                        part_res = part[response_eot_id+len(eot)+1:]
                        part_res_len = len(tokenizer_video_token(part_res, self.tokenizer))
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

    def _set_same_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        set_seed(seed)
    
    def preprocess_input(self, input_data: SignLlavaInput):
        visual_features = {}
        for input_type in INPUT_TYPES:
            if input_type!="pose" and eval(f"input_data.{input_type}_features") is not None:
                vf = torch.tensor(eval(f"input_data.{input_type}_features")).to(torch.bfloat16)
                visual_features[input_type] = vf 
        
        prompt = input_data.prompt
        src = {}
        video_token = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN
        src['conversations'] = [{'from': 'human', 
                'value':video_token+'\n'+ prompt},
                {'from': 'gpt',
                'value':"placeholder"}]
        data_dict = self.preprocess_llama_3(src)
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]
        input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids.unsqueeze(0),
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels.unsqueeze(0),
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.model_max_length].to(self.model.device)
        labels = labels[:, :self.model_max_length].to(self.model.device)
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['visual_features'] = visual_features
        video_sep = DEFAULT_VIDEO_END_TOKEN+DEFAULT_VIDEO_START_TOKEN
        data_dict['video_sep_ids'] = tokenizer_video_token(video_sep, self.tokenizer, return_tensors='pt')
        return data_dict

    def run_inference(self, input_data: SignLlavaInput) -> SignLlavaOutput:
        data_dict = self.preprocess_input(input_data)
        generate_kwargs = dataclasses.asdict(input_data.generation_config)
        with torch.inference_mode():
            visual_embeddings_out = {}
            output_dict = self.model.generate(
                inputs = data_dict['input_ids'],
                labels = data_dict['labels'], # neccessary placeholder
                visual_features = [data_dict['visual_features']],
                video_sep_ids = [data_dict['video_sep_ids']],
                pad_token_id = self.tokenizer.unk_token_id,
                output_scores = True,
                return_dict_in_generate=True,
                forced_bos_token_id = self.tokenizer.encode("\n\n")[1:],
                decoder_start_token_id = self.tokenizer.encode("\n\n")[1:],
                bos_token_id = self.tokenizer.encode("\n\n")[1],
                visual_embeddings_out = visual_embeddings_out,
                **generate_kwargs
            )
            output_ids = output_dict['sequences']
            outputs = self.tokenizer.batch_decode(output_ids, 
                skip_special_tokens=self.config['GenerateArguments']['skip_special_tokens'])[0].strip()
        return SignLlavaOutput(
            output=outputs,
            sign2vec_embeddings=visual_embeddings_out['sign2vec'].cpu().data.to(torch.float32).numpy(),
            mae_embeddings=visual_embeddings_out['mae'].cpu().data.to(torch.float32).numpy(),
            dino_embeddings=visual_embeddings_out['dino'].cpu().data.to(torch.float32).numpy()
        )
