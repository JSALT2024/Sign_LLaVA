import argparse

import sacrebleu

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    return parser.parse_args()