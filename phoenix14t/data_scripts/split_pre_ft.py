# Copyright (c) 2024 Xuan Zhang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import json

def get_args():
    parser = argparse.ArgumentParser(description="Split train set into pretraining and fine-tuning set.")
    parser.add_argument("--pkl", default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/S3D_features.train.pkl", 
        help="The pickle file.")
    parser.add_argument("--json", default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/anno.train.json",
        help="The json file.")
    parser.add_argument("--num-examples-in-pretraining", '-n', default=5000,
        help="The number of examples in the pretraining dataset.")
    parser.add_argument("--outdir", default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data", 
        help="The directory to output.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.pkl, 'rb') as f:
        pkl_data = pickle.load(f)
    with open(args.json) as f:
        json_data = json.load(f)
    n = args.num_examples_in_pretraining
    json_pre = json_data[:n]
    json_ft = json_data[n:]
    pkl_pre = {}
    pkl_ft = {}
    for ij in json_pre:
        name = ij['id']
        pkl_pre[name] = pkl_data[name]
    for ij in json_ft:
        name = ij['id']
        pkl_ft[name] = pkl_data[name]
    with open(os.path.join(args.outdir, "S3D_features.pretrain.pkl"), 'wb') as f:
        pickle.dump(pkl_pre, f)
    with open(os.path.join(args.outdir, "S3D_features.finetune.pkl"), 'wb') as f:
        pickle.dump(pkl_ft, f)
    with open(os.path.join(args.outdir, "anno.pretrain.json"), 'w') as f:
        json.dump(json_pre, f)
    with open(os.path.join(args.outdir, "anno.finetune.raw.json"), 'w') as f:
        json.dump(json_ft, f)

if __name__ == "__main__":
    main()