import os
import random
from enum import Enum
from sys import platform

import numpy as np
import tensorflow_datasets as tfds
import torch
from tqdm import tqdm

from modules.ExtendedMosesTokenizer import ExtendedMosesTokenizer


class Splits(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3


class Corpus(object):
    def __init__(self, tokenizer, use_moses=True):
        self.tokenizer = tokenizer
        self.moses_tokenizer = ExtendedMosesTokenizer()
        self.use_moses = use_moses

        self.data = {
            Splits.TRAIN: [],
            Splits.VALID: [],
            Splits.TEST: []
        }
        self.summary_mask = {
            Splits.TRAIN: [],
            Splits.VALID: [],
            Splits.TEST: []
        }

    def get_data_flat(self, split: Splits):
        if isinstance(self.data[split], list):
            return torch.cat(self.data[split])
        else:
            return self.data[split]

    def get_summary_mask_flat(self, split: Splits):
        if isinstance(self.data[split], list):
            return torch.cat(self.summary_mask[split])
        else:
            return self.summary_mask[split]

    def set_use_moses(self, use_moses):
        self.use_moses = use_moses

    def encode_text(self, text: str):
        if self.use_moses:
            tokens = self.moses_tokenizer.tokenize(text)
            return self.tokenizer.encode(tokens)
        else:
            return self.tokenizer.encode(text, add_space_before_punct_symbol=True)

    def decode_ids(self, ids) -> str:
        if self.use_moses:
            text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            return self.moses_tokenizer.detokenize(text)
        else:
            return self.tokenizer.decode(ids)

    def split_example_in_src_tgt(self, idx: int, split: Splits, return_masks=False):
        example = self.data[split][idx]
        mask = self.summary_mask[split][idx]
        cls_idx = (example == self.tokenizer.cls_token_id).nonzero()

        src = example[:cls_idx]
        tgt = example[cls_idx+1:-1]     # +1 for CLS, :-1 for EOS

        if return_masks:
            src_mask = mask[:cls_idx]
            tgt_mask = mask[cls_idx+1:-1]
            return src, tgt, src_mask, tgt_mask
        else:
            return src, tgt

    def encode_dataset(self, dataset, split: Splits, flatten=False):
        encoded = []
        summary_mask = []
        for idx, sample in tqdm(enumerate(dataset), desc='Encoding articles', unit=' examples', mininterval=(2*60)):
            src, tgt = sample['article'], sample['highlights']
            src, tgt = bytes.decode(src.numpy()), bytes.decode(tgt.numpy())
            tgt = tgt.replace("\n", " ")

            # Mosestokenizer
            if self.use_moses:
                src_tokens = self.moses_tokenizer.tokenize(src)
                tgt_tokens = self.moses_tokenizer.tokenize(tgt)
            else:
                src_tokens = src
                tgt_tokens = tgt

            add_spaces = not self.use_moses
            src_ids = self.tokenizer.encode(src_tokens, add_space_before_punct_symbol=add_spaces)
            tgt_ids = self.tokenizer.encode(tgt_tokens, add_space_before_punct_symbol=add_spaces)

            # src + " <cls> " + tgt + " <eos>"
            sample_ids = src_ids + [self.tokenizer.cls_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
            sample_ids = torch.tensor(sample_ids)

            # construct summary mask (1 if token is from summary, -1 for EOS, 0 otherwise)
            src_mask = torch.zeros(len(src_ids) + 1)
            tgt_mask = torch.ones(len(tgt_ids) + 1)
            tgt_mask[-1] = -1    # EOS is not part of summary
            mask = torch.cat((src_mask, tgt_mask))

            assert sample_ids.size() == mask.size(), \
                f"Tokenized sample has to be of same shape as it's corresponding mask"

            summary_mask.append(mask)
            encoded.append(sample_ids)

        if flatten:
            encoded = torch.cat(encoded)

        self.data[split] = encoded
        self.summary_mask[split] = summary_mask

    def truncate_all_examples(self, max_src_len=750, max_tgt_len=100):
        for split in list(Splits):
            truncated = []
            for idx, example in enumerate(self.data[split]):
                cls_index = (example == self.tokenizer.cls_token_id).nonzero()

                # Truncate example
                trunc_example = truncate_example(example, cls_index, max_src_len, max_tgt_len)
                truncated.append(trunc_example)

                # Truncate mask
                trunc_mask = truncate_example(self.summary_mask[split][idx], cls_index, max_src_len, max_tgt_len)
                self.summary_mask[split][idx] = trunc_mask

            self.data[split] = truncated


def split_example(example: torch.Tensor, delimiter):
    cls_idx = (example == delimiter).nonzero().squeeze()
    src = example[:cls_idx]
    tgt = example[cls_idx + 1:-1]  # +1 for CLS, :-1 for EOS
    return src, tgt


def truncate_example(example, cls_idx, max_src_len, max_tgt_len):
    src = example[:cls_idx]
    tgt = example[cls_idx:-1]

    src = src[:max_src_len]
    tgt = tgt[:(max_tgt_len + 1)]       # +1 for CLS token

    return torch.cat((src, tgt, example[-1].unsqueeze(0)))


def split_to_tfsplit(split: Splits):
    if split == Splits.TRAIN:
        return 'train'
    elif split == Splits.VALID:
        return 'validation'
    elif split == Splits.TEST:
        return 'test'


def get_cnndm_dataset(split: Splits, corpus_pct, return_info=False):
    if platform == "win32":
        cnndm_builder = tfds.builder("cnn_dailymail", data_dir="C:/Users/weing/tensorflow_datasets")
    else:
        cnndm_builder = tfds.builder("cnn_dailymail")
    cnndm_builder.download_and_prepare()

    split_str = f"{split_to_tfsplit(split)}[:{corpus_pct}%]"

    if return_info:
        return cnndm_builder.as_dataset(split=split_str), cnndm_builder.info

    return cnndm_builder.as_dataset(split=split_str)


def get_random_set(pct, interval):
    sets = [x for x in range(0, 101, pct)]
    fac = len(sets) // interval
    rnd = random.randint(0,interval-1)
    idx = rnd * fac
    return sets[idx], sets[idx+1]


def write_system_summaries(data_dir, corpus_pct, split=Splits.TEST):
    cnndm = get_cnndm_dataset(split, corpus_pct)
    sys_sum_path = os.path.join(data_dir, 'system_summaries')
    if not os.path.exists(sys_sum_path):
        os.mkdir(sys_sum_path)

    for idx, sample in enumerate(cnndm):
        filepath = os.path.join(sys_sum_path, f'summary.{str(idx+1).zfill(4)}.txt')
        with open(filepath, 'w', encoding='utf-8') as file:
            tgt = sample['highlights']
            # src, tgt = tf.strings.as_string(src), tf.strings.as_string(tgt)
            tgt = bytes.decode(tgt.numpy())
            # src = bytes.decode(src.numpy(), errors='replace', encoding='utf-8')
            # tgt = bytes.decode(tgt.numpy(), errors='replace', encoding='utf-8')
            tgt = tgt.replace("\n", " ")
            file.write(tgt)


def read_generated_summaries(path):
    assert os.path.exists(path)
    with open(path, "r") as file:
        lines = file.readlines()

    counter = 0
    summaries = []
    for t in lines:
        if t.strip() == "":
            continue
        num, line = t.split(":\t ")
        line = line.strip()

        assert int(num) == counter, f"Expected line number {counter}, but it was {num}"
        assert len(line) > 1, f"Line to short: {line}"
        summaries.append(line)
        counter += 1

    print(f"Done reading {counter} summaries from {path}.")
    return summaries


# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FROM: src/transformers/tokenization_transfo_xl.py
""" Tokenization classes for Transformer XL model.
    Adapted from https://github.com/kimiyoung/transformer-xl.
"""
class LMOrderedIteratorHuggFace(object):
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None, summary_mask=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.summary_mask = None

        self.device = device

        if summary_mask is not None:
            assert data.size(0) == summary_mask.size(0)

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous()

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

        # Reshape summary mask in the same way as the normal data
        if summary_mask is not None:
            summary_mask = summary_mask[:self.n_step * bsz]
            self.summary_mask = summary_mask.view(bsz, -1).t().contiguous()

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        data_out = data.transpose(0, 1).contiguous().to(self.device)
        target_out = target.transpose(0, 1).contiguous().to(self.device)

        if self.summary_mask is not None:
            tgt_summary_mask = self.summary_mask[beg_idx:end_idx]
            tgt_summary_mask = tgt_summary_mask.transpose(0, 1).contiguous()
        else:
            tgt_summary_mask = None

        return data_out, target_out, seq_len, tgt_summary_mask

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len, _ = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


