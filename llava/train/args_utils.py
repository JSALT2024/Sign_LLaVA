from dataclasses import dataclass, field
from typing import Optional, Any
import torch

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta/Meta-Llama-3-8B-Instruct")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_embed_tokens: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=".")
    bf16: bool = field(default=True)
    report_to: Optional[str] = field(default="wandb")
    # gradient_accumulation_steps: Optional[int] = field(default=1)
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


def prepare_bnb_args(training_args: Any, compute_dtype: torch.dtype, skip_modules: dict):
    bnb_model_from_pretrained_args = {}

    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map="auto",
            device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int4_skip_modules=skip_modules,
                llm_int8_skip_modules=skip_modules,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,  # must be false if --bf True
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    return bnb_model_from_pretrained_args
