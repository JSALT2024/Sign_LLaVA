# Copyright (c) 2024 Xuan Zhang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import json
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Format PHOENIX14T data into LLaVA inputs.")
    parser.add_argument("--indir", default="/exp/xzhang/slt/slr_handshape/best_model/s3d_features", 
        help="The directory to the original PHOENIX14T data with S3D features.")
    parser.add_argument("--dname", default="PHOENIX14T_HS_S3Dfeatures",
        help="The prefix of the PHOENIX14T data files.")
    parser.add_argument("--outdir", default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data", 
        help="The directory to the formatted data.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    for split in ['train', 'dev', 'test']:
        ori_dfile = os.path.join(args.indir, args.dname + "."+split+".pkl")
        with open(ori_dfile, 'rb')  as f:
            data = pickle.load(f)
        """
        {
            'name': ['test/25October_2010_Monday_tagesschau-17'], 
            'gloss': ['REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN'], 
            'text': ['regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar .'], 
            's3d_params': tensor(T//4, 832),
            'alignment': [None], 
            'num_frames': [181], #T
            'signers': ['Signer01'], 
            'handshape-right': [[['5'], ['index_flex', '5'], ['5'], ['b'], ['b'], ['5'], ['s'], ['5'], ['index'], ['s'], ['v']]], 
            'handshape-left': [[['5'], ['5', 'ae', '5'], [], ['ital_nothumb'], ['b'], ['5'], ['s'], [], ['index'], ['s'], []]]}
            }
        """
        # 1. s3d_dic: {"name": s3d_params (T//4, 832)} -> pickle
        s3d_dic_file = os.path.join(args.outdir, "S3D_features.{}.pkl".format(split))
        # 2. annotation: {'id': name, 'image': name, 
        #                 'conversations':
        #                 [{'from':'human', 'value':'<image>\n Translate the German sign language video into German.'},
        #                  {'from':'gpt', 'value':text}]}
        anno_file = os.path.join(args.outdir, "anno.{}.json".format(split))
        s3d_dic = {}
        annos = []
        print("Dumping {} set...".format(split))
        for ditem in tqdm(data):
            name = ditem['name'][0]
            text = ditem['text'][0]
            s3d_params = ditem['s3d_params']
            s3d_dic[name] = s3d_params
            anno = {}
            anno['id'] = name
            anno['image'] = name
            anno['conversations'] = [{'from': 'human', 
                'value': '<image>\n Translate the German sign language video into German.'},
                {'from': 'gpt', 'value': text}]
            annos.append(anno)
        with open(s3d_dic_file, 'wb') as f:
            pickle.dump(s3d_dic, f)
        with open(anno_file, 'w') as f:
            json.dump(annos, f, indent=4)

if __name__ == "__main__":
    main()
