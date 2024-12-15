import os
from PIL import Image
import torch
from torch import nn
import numpy as np
from typing import List, Union
from torchvision import transforms
from abc import ABC, abstractmethod

from llava.encoders import mae_models_vit, get_keypoints
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
    ) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        predictions = []
        batches = self.create_batches(images, self.batch_size)
        for batch in batches:
            images = [self.normalize_image(image, self.image_size) for image in batch]
            images = torch.stack(images).to(device)

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
    ) -> torch.Tensor:
        normalized_keypoints = self.normalize_keypoints(input_predictions)
        normalized_keypoints = torch.tensor(normalized_keypoints).float().to(device)

        predictions = []
        batches = self.create_batches(normalized_keypoints, self.batch_size)
        for batch in batches:
            prediction = self.model(batch)
            predictions.append(prediction)
        predictions = torch.cat(predictions, 0)

        return predictions
