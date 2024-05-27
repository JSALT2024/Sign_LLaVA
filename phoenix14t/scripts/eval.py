from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from run_llava import eval_model

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on PHOENIX14T datasets (test set).")
    parser.add_argument("--checkpoint", type=str, 
                        default="/exp/xzhang/slt/jsalt2024/LLaVA/checkpoints/llava-vicuna-v1-3-7b-finetune_lora_phoenix14t_s3d", 
                        help="The directory to finetuned model checkpoints.")
    parser.add_argument("--json-data", type=str, help="The json file.",
                        default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/anno.test.json")
    parser.add_argument("--pkl-data", type=str, help="The pickle file.",
                        default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/S3D_features.test.pkl")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_path = args.checkpoint
    model_base = "lmsys/vicuna-7b-v1.3"
    args = type('Args', (), {
    "model_path": model_path,
    "model_base": model_base,
    "model_name": get_model_name_from_path(model_path),
    "anno": args.json_data,
    "s3d": args.pkl_data,
    "conv_mode": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
    })()
    eval_model(args)
if __name__ == "__main__":
    main()