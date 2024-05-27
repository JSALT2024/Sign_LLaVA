import argparse
import os
import json
import sacrebleu

def get_args():
    parser = argparse.ArgumentParser(description="Get BLEU scores.")
    parser.add_argument("--json-data", type=str, help="The json file.",
                        default="/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/data/anno.test.json")
    parser.add_argument("--predictions", type=str, help="The predictions.",
                        default="/exp/xzhang/slt/jsalt2024/LLaVA/checkpoints/llava-vicuna-v1-3-7b-finetune_lora_phoenix14t_s3d/predictions.json")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.json_data) as f:
        data_lst = json.load(f)
    with open(args.predictions) as f:
        pred_dic = json.load(f)
    preds = []
    refs = []
    for item in data_lst:
        name = item['id']
        ref = item['conversations'][1]['value']
        refs.append(ref)
        pred = pred_dic[name]
        preds.append(pred)
        print(ref, pred)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print(bleu)

if __name__ == "__main__":
    main()