# Copyright (c) 2024 Xuan Zhang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import json
import random

def get_args():
    parser = argparse.ArgumentParser(description="Set prompt variants for fine-tuning train set.")
    parser.add_argument("--injson", default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/anno.{}.raw.json",
        help="The json input.")
    parser.add_argument("--outjson", default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/anno.{}.json", 
        help="The json output.")
    args = parser.parse_args()
    return args

prompts = [
    "Translate the German sign language video about weather forecasts into German.",
    "Translate the German sign language video into German.",
    "Convert the content of a German sign language video related to weather forecasts into written German.",
    "Convert a German sign language video into written German.",
    "Transcribe the German sign language video on weather forecasting into German text.",
    "Transcribe a German sign language video into German text.",
    "Interpret the weather-related German sign language video into German.",
    "Interpret a German sign language video into German.",
    "Render the German sign language content about weather forecasts into German.",
    "Render a German sign language video into German."
    ]

def main():
    args = get_args()
    for split in ["finetune", "dev", "test"]:
        with open(args.injson.format(split)) as f:
            json_data = json.load(f)
        json_out = []
        for ij in json_data:
            ij['conversations'][0]['value'] = random.choice(prompts)
            json_out.append(ij)
        with open(args.outjson.format(split), 'w') as f:
            json.dump(json_out, f)

if __name__ == "__main__":
    main()