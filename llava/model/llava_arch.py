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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from llava.constants import *

class SignLlavaProjector: # adapted from LlavaMetaModel
    def __init__(self, sign_model_args, sign_data_args, config):
        super().__init__()
        self.config = config
        self.sign_model_args = sign_model_args
        projector_configs = sign_model_args['projectors']
        for input_type in INPUT_TYPES:
            projector_name = "{}_projector".format(input_type)
            projector_args = projector_configs[input_type]
            if sign_data_args['visual_features'][input_type]["enable_input"]:
                projector_type = projector_args['projector_type']
                mm_hidden_size = projector_args['dim']
                hidden_size = self.config.hidden_size
                exec(f"self.{projector_name}=build_vision_projector(projector_type, mm_hidden_size, hidden_size)")
            else:
                exec(f"self.{projector_name}=None")

    def initialize_projectors(self): # adapted from initialize_vision_modules, it seems it is never called! 
        projector_configs = self.sign_model_args['projectors']
        '''
        projector_names = []
        for input_type in INPUT_TYPES:
            projector_name = "{}_projector".format(input_type)
            projector_args = projector_configs[input_type]
            if eval(f"self.{projector_name}") is not None:
                projector_names.append(projector_name)
                # In case it is frozen by LoRA
                for p in eval(f"self.{projector_name}.parameters()"):
                    p.requires_grad = True
        '''
        projector_names = []
        for input_type in INPUT_TYPES:
            projector_name = "{}_projector".format(input_type)
            projector_args = projector_configs[input_type]
            if eval(f"self.{projector_name}") is not None:
                projector_names.append(projector_name)
        # load weights from a pretrained checkpoint
        pretrained_ckpt = projector_configs.get("pretrained_projector_ckpt", None)
        if pretrained_ckpt is not None:
            projector_weights = torch.load(pretrained_ckpt, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            for projector_name in projector_names:
                eval(f"self.{projector_name}").load_state_dict(get_w(projector_weights['module'], f'{projector_name}'))
                print(f"[SignLLaVA]: Loaded the pretrained weights for {projector_name} from {pretrained_ckpt}.")

class SignLlavaForCausalLM(ABC): # adapted from LlavaMetaForCausalLM(ABC)
    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_and_labels(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        video_sep_ids,
        visual_features):
        # Inputs are batched!
        # visual_features: a list of dictionaries: {"sign2vec": torch.tensor}

        # Step 1. Get projected visual features.
        # Step 2. Separate the inputs to [tokens before the visual features] <video> [tokens after the visual features].
        # Step 3. Get embeddings.
        # Step 4. Truncate the input to tokenizer_model_max_length
        # Step 5. Pad the input to the same length in a batch.
        if visual_features == [] or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # Step 1. Get projected visual features.
        projected_visual_features = []
        for vf_dict in visual_features:
            projected_vf_dict = {}
            for key in vf_dict:
                visual_feature = vf_dict[key].to(input_ids.device)
                projected_vf_dict[key] = eval(f"self.get_model().{key}_projector(visual_feature)")
            projected_visual_features.append(projected_vf_dict)
        del visual_features
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        # Initialize attention_mask, position_ids, labels
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Step 2. Separate the inputs to [tokens before the visual features] <video> [tokens after the visual features].
            video_token_indices = [-1] + torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # image_token_indices: [-1, <video> loc, len(input_ids)]
            cur_input_ids_noim = [] # noim: no image, two parts: half before <video>, half after
            cur_labels_noim = [] # two parts: half before <video>, half after
            cur_labels = labels[batch_idx]
            for i in range(len(video_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[video_token_indices[i]+1:video_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[video_token_indices[i]+1:video_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # Step 3. Get embeddings.
            # Get text embeddings.
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # cur_input_embeds_no_im: two parts: half before <video>, half after
            cur_new_input_embeds = []
            cur_new_labels = []

            # <video_start><video><video_end> -> <video_start>rep1<video_end><video_start>rep2<video_end>
            cur_new_input_embeds.append(cur_input_embeds_no_im[0])
            cur_new_labels.append(cur_labels_noim[0])
            cur_projected_feature_dict = projected_visual_features[batch_idx]
            video_sep_embeds = self.get_model().embed_tokens(video_sep_ids[0].to(input_ids[0].device))
            visual_embeddings = []
            for input_type in cur_projected_feature_dict:
                if visual_embeddings != []:
                    visual_embeddings.append(video_sep_embeds)
                visual_embeddings.append(cur_projected_feature_dict[input_type])
            cur_visual_embeddings = torch.cat(visual_embeddings)
            cur_new_input_embeds.append(cur_visual_embeddings)
            cur_new_labels.append(torch.full((cur_visual_embeddings.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds.append(cur_input_embeds_no_im[1])
            cur_new_labels.append(cur_labels_noim[1])
            """
            cur_new_labels:
            [tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100], device='cuda:0'), tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            device='cuda:0'), tensor([  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                -100,   -100,   -100,   -100,   -100,  59270,   9909,    268,   5409,
                531,   1560,  88891,   7367,  19649,   1560,  28398,  18268,   5817,
                28196,  14230,  61397,    737,  48734,  11285,    268, 127318,   6915,
                5568,  28699,    455,    301,  12666,  72534,   2073,  71751,    662,
                128009], device='cuda:0')]
            """
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        # padding to max length in the batch
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, projected_visual_features[0]

    def initialize_vision_tokenizer(self, model_args, tokenizer, sign_model_args):
        num_new_tokens = tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if model_args.tune_mm_mlp_adapter:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False

        pretrained_ckpt = sign_model_args['projectors'].get("pretrained_projector_ckpt", None)
        if pretrained_ckpt:
            pretrained_weights = torch.load(pretrained_ckpt, map_location='cpu')
            print("[SignLLaVA]: Loaded the pretrained weights for embed_tokens from {}.".format(sign_model_args['projectors']['pretrained_projector_ckpt']))
            embed_tokens_weight = pretrained_weights['module']['model.embed_tokens.weight']
            assert num_new_tokens == 2
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
