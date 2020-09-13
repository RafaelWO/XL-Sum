import argparse
import os

import nltk
from tqdm import tqdm

import modules.utils as utils
from modules.data_builder import build_corpus
from modules.data_utils import Splits, split_example

parser = argparse.ArgumentParser()
utils.add_corpus_args(parser)
args = parser.parse_args()
utils.create_logger(args)

corpus = build_corpus(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
save_dir = 'trainings/lead_3/'

dataset = corpus.data[Splits.TEST]
lead3_summaries = []
targets = []
for idx, sample in enumerate(tqdm(dataset, desc='Generating lead summary', unit='examples')):
    src, tgt = split_example(sample, corpus.tokenizer.cls_token_id)
    src = corpus.decode_ids(src)
    tgt = corpus.decode_ids(tgt)

    lead3 = nltk.sent_tokenize(src)
    lead3_summaries.append(lead3[:3])
    targets.append(tgt)

with open(save_dir + "generated_summaries_eval.txt", "w") as f:
    for i, summary in enumerate(lead3_summaries):
        f.write(f"{i}:\t {' '.join(summary)}\n\n")

with open(save_dir + "target_summaries_new.txt", "w") as f:
    for i, summary in enumerate(targets):
        f.write(f"{i}:\t {summary}\n\n")