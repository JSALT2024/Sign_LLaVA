import os
from PIL import Image
import torch
from torch import nn
import numpy as np
from typing import List, Union, Optional
from torchvision import transforms
from abc import ABC, abstractmethod

from llava.encoders import mae_models_vit, get_keypoints, get_local_crops
from .multimodal_projector.builder import build_vision_projector


class Encoder(ABC):
    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        ...

    @abstractmethod
    def initialize_model(self, checkpoint_path: str, model_key: str = "model") -> None:
        ...

    def require_grad(self, require: bool) -> None:
        for parameters in self.model.parameters():
            parameters.requires_grad = require

    @staticmethod
    def create_batches(input_list, chunk_size):
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

    def forward(
        self,
        images: Union[List[np.array], np.array],
        input_predictions: dict,
        device: torch.device,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        ...


class MAEEncoder(Encoder, nn.Module):
    def __init__(self, arch: str, output_dim: int, projector: str, **kwargs):
        super().__init__()
        self._mae_model = mae_models_vit.__dict__[arch](global_pool=False)
        self._projection_layer = build_vision_projector(projector, self._mae_model.embed_dim, output_dim)

        self._mae_model = nn.Sequential(
            self._mae_model,
            self._projection_layer,
        )

        self.batch_size = kwargs.get('batch_size', 16)
        self.image_size = kwargs.get('image_size', (224, 224))
        self.image_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def model(self) -> nn.Module:
        return self._mae_model

    @property
    def encoder_model(self) -> nn.Module:
        return self._mae_model[0]

    def initialize_model(self, checkpoint_path: str, model_key: str = "model", **kwargs) -> None:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"[MAEEncoder]: no checkpoint at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = self.encoder_model.load_state_dict(checkpoint[model_key], strict=False)

        print(f"[MAEEncoder]: Missing head and unexpected decoder keys are expected.")
        print(f"[MAEEncoder]: Load checkpoint message: {msg}")

    def normalize_image(self, image_bgr: np.ndarray, image_size: tuple = (224, 224)):
        image_rgb = image_bgr[..., ::-1]
        image = Image.fromarray(np.uint8(image_rgb)).convert('RGB')
        image = image.resize(image_size)
        image = self.transform(image)
        return image

    def forward(
        self,
        images: Union[List[np.array], np.array],
        input_predictions: dict,
        device: torch.device,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        predictions = []
        batches = self.create_batches(images, self.batch_size)
        for batch in batches:
            images = [self.normalize_image(image, self.image_size) for image in batch]
            images = torch.stack(images).to(device)
            if dtype is not None:
                images = images.type(dtype)

            prediction = self.model(images)
            predictions.append(prediction)
        predictions = torch.cat(predictions, 0)

        return predictions


class PoseEncoder(Encoder, nn.Module):
    def __init__(self, input_dim: int, output_dim: int, projector: str, **kwargs):
        super().__init__()
        self._pose_model = build_vision_projector(projector, input_dim, output_dim)

        self.batch_size = kwargs.get('batch_size', 16)
        self.normalization_methods = kwargs.get('normalization_methods',
                                                [
                                                    "global-pose_landmarks",
                                                    "local-right_hand_landmarks",
                                                    "local-left_hand_landmarks",
                                                    "local-face_landmarks"
                                                ])

        # magical variables
        self.data_key = "cropped_keypoints"
        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
            93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
            282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
        ]

    @property
    def model(self) -> nn.Module:
        return self._pose_model

    @property
    def encoder_model(self) -> nn.Module:
        return self._pose_model

    def initialize_model(self, checkpoint_path: str, model_key: str = "model", **kwargs) -> None:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"[PoseEncoder]: no checkpoint at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = self.encoder_model.load_state_dict(checkpoint[model_key], strict=False)

        print(f"[PoseEncoder]: Load checkpoint message: {msg}")

    def normalize_keypoints(self, keypoints: dict):
        normalized_keypoints = get_keypoints(
            keypoints,
            self.data_key,
            self.face_landmarks,
            self.normalization_methods
        )
        return normalized_keypoints

    def forward(
        self,
        images: Union[List[np.array], np.array],
        input_predictions: dict,
        device: torch.device,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        normalized_keypoints = self.normalize_keypoints(input_predictions)
        normalized_keypoints = torch.tensor(normalized_keypoints).float().to(device)

        predictions = []
        batches = self.create_batches(normalized_keypoints, self.batch_size)
        for batch in batches:
            if dtype is not None:
                batch = batch.type(dtype)
            prediction = self.model(batch)
            predictions.append(prediction)
        predictions = torch.cat(predictions, 0)

        return predictions


class DINOEncoder(Encoder, nn.Module):
    def __init__(self, arch: str, output_dim: int, projector: str, **kwargs):
        super().__init__()
        self._dino_face_model = torch.hub.load('facebookresearch/dinov2', arch, pretrained=False)
        self._dino_hand_model = torch.hub.load('facebookresearch/dinov2', arch, pretrained=False)
        self._dino_face_model.pos_embed = nn.Parameter(torch.zeros(1, 257, self._dino_face_model.embed_dim))
        self._dino_hand_model.pos_embed = nn.Parameter(torch.zeros(1, 257, self._dino_hand_model.embed_dim))

        input_dim = self._dino_face_model.embed_dim + (self._dino_hand_model.embed_dim * 2)
        self._projection_layer = build_vision_projector(projector, input_dim, output_dim)

        self._dino_model = nn.ModuleList([
            nn.ModuleDict({
                "face_model": self._dino_face_model,
                "hand_model": self._dino_hand_model
            }),
            self._projection_layer
        ])

        self.batch_size = kwargs.get('batch_size', 16)
        self.image_size = kwargs.get('image_size', (224, 224))
        self.image_size = (self.image_size, self.image_size) if isinstance(self.image_size,
                                                                           int) else self.image_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def model(self) -> nn.Module:
        return self._dino_model

    @property
    def encoder_model(self) -> nn.Module:
        return self._dino_model[0]

    @staticmethod
    def _rename_parameters(checkpoint):
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if 'dino_head' in key:
                continue
            else:
                new_key = key.replace('backbone.', '')
                new_checkpoint[new_key] = value
        return new_checkpoint

    def initialize_model(self, checkpoint_path: Union[list, str], model_key: str = "teacher", **kwargs) -> None:
        face_checkpoint_path = checkpoint_path[0]
        hand_checkpoint_path = checkpoint_path[1]
        if not os.path.exists(face_checkpoint_path):
            raise ValueError(f"[DINO2Encoder]: no checkpoint at {face_checkpoint_path}")
        if not os.path.exists(hand_checkpoint_path):
            raise ValueError(f"[DINO2Encoder]: no checkpoint at {hand_checkpoint_path}")

        face_checkpoint = torch.load(face_checkpoint_path, map_location='cpu')
        face_checkpoint = self._rename_parameters(face_checkpoint)
        msg = self.encoder_model.load_state_dict(face_checkpoint[model_key], strict=False)
        print(f"[DINO2Encoder]: Load face_checkpoint message: {msg}")

        hand_checkpoint = torch.load(hand_checkpoint_path, map_location='cpu')
        hand_checkpoint = self._rename_parameters(hand_checkpoint)
        msg = self.encoder_model.load_state_dict(hand_checkpoint[model_key], strict=False)
        print(f"[DINO2Encoder]: Load hand_checkpoint message: {msg}")

    def normalize_image(self, image_bgr: np.ndarray, image_size: tuple = (224, 224)):
        image_rgb = image_bgr[..., ::-1]
        image = Image.fromarray(np.uint8(image_rgb)).convert('RGB')
        image = self.transform(image)
        return image

    def _normalize_batch_images(self, images: List[np.ndarray], device: torch.device) -> torch.tensor:
        images = [self.normalize_image(image, self.image_size) for image in images]
        images = torch.stack(images).to(device)
        return images

    def forward(
        self,
        images: Union[List[np.array], np.array],
        input_predictions: dict,
        device: torch.device,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        predictions = []
        # prepare batches
        image_batches = self.create_batches(images, self.batch_size)
        bbox_batches = {name: self.create_batches(input_predictions[name], self.batch_size)
                        for name in input_predictions}
        _bbox_batches = []
        for bidx in range(np.ceil(len(images) / self.batch_size).astype(int)):
            _batch = {}
            for name in bbox_batches:
                _batch[name] = bbox_batches[name][bidx]
            _bbox_batches.append(_batch)
        bbox_batches = _bbox_batches

        # prepare models
        models, projection_layer = self.model
        face_model = models["face_model"]
        hand_model = models["hand_model"]

        # run batch predictions
        for image_batch, bbox_batch in zip(image_batches, bbox_batches):
            images_face, images_left, images_right = get_local_crops(image_batch, bbox_batch)
            images_face = self._normalize_batch_images(images_face, device)
            images_left = self._normalize_batch_images(images_left, device)
            images_right = self._normalize_batch_images(images_right, device)
            if dtype is not None:
                images_face = images_face.type(dtype)
                images_left = images_left.type(dtype)
                images_right = images_right.type(dtype)

            face_features = face_model(images_face)
            left_features = hand_model(images_left)
            right_features = hand_model(images_right)
            features = torch.cat([face_features, left_features, right_features], 1)
            prediction = projection_layer(features)
            predictions.append(prediction)
        predictions = torch.cat(predictions, 0)

        return predictions
