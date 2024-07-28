import sacrebleu
import json

with open("generation.test.json", "r") as f:
    res = json.load(f)
gen_lines = []
ref_lines = []
for video_name in res:
    for clip_name in res[video_name]:
        if clip_name == "clip_order":
            continue
        clip_dict = res[video_name][clip_name]
        gen_lines.append(clip_dict["hypothesis"])
        ref_lines.append(clip_dict["translation"])
        sent_bleu = sacrebleu.sentence_bleu(clip_dict["hypothesis"], [clip_dict["translation"]])
        print("reference:", clip_dict["translation"])
        print("hypothesis:", clip_dict["hypothesis"])
        print("sentence bleu:", sent_bleu)
        print()
bleu = sacrebleu.corpus_bleu(gen_lines, [ref_lines])
print(bleu)