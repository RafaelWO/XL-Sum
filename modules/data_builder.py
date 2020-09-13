import sys
if __name__ == "__main__":
    sys.path.insert(0, "../")

import logging
import os
from sys import platform
import torch
from transformers import TransfoXLTokenizer
from modules.data_utils import Corpus, Splits, get_cnndm_dataset
import modules.utils as utils
import time


def build_corpus(args):
    logger = logging.getLogger(args.logger_name)
    cls_add_idx = 20000 if args.mode != "base" else None

    # Load tokenizer
    old = "_old" if args.mode == "base" else ""
    tokenizer_path = os.path.join(args.data_dir, f"tokenizer{old}")
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")) and not args.rebuild_corpus:
        tokenizer = TransfoXLTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        tokenizer = add_cls_token_to_tokenizer(tokenizer, args.logger_name, add_idx=cls_add_idx)
        tokenizer.save_pretrained(tokenizer_path)

    assert tokenizer.cls_token_id == cls_add_idx or tokenizer.cls_token_id == len(tokenizer) - 1

    if args.mode == "moses":
        postfix = "_moses"
    elif args.mode != "base":
        postfix = "_move"
    else:
        postfix = ""
    cached_file = os.path.join(args.data_dir, f"data_cache_{args.corpus_pct}pct{postfix}.pt")
    if os.path.exists(cached_file) and not args.rebuild_corpus:
        # Load cached dataset
        logger.info(f"Loading cached dataset {cached_file}")
        corpus = torch.load(cached_file)
        if getattr(corpus, "use_moses", -1) == -1:
            corpus.set_use_moses(args.mode == "moses")
        for split in Splits:
            if len(corpus.data[split]) == 0:
                logger.warning(f"The split {split} does not contain data!")
    else:
        # Load data
        logger.info(f"Selecting {args.corpus_pct}% of the CNN/DM dataset as corpus")
        cnndm_train = get_cnndm_dataset(Splits.TRAIN, args.corpus_pct)
        cnndm_valid = get_cnndm_dataset(Splits.VALID, args.corpus_pct)
        cnndm_test = get_cnndm_dataset(Splits.TEST, args.corpus_pct)

        logger.info("=" * 100)

        # Build corpus
        corpus = Corpus(tokenizer, args.mode == "moses")
        start_time = time.time()
        corpus.encode_dataset(cnndm_train, split=Splits.TRAIN)
        corpus.encode_dataset(cnndm_valid, split=Splits.VALID)
        corpus.encode_dataset(cnndm_test, split=Splits.TEST)
        elapsed = time.time() - start_time
        logger.info(f"Elapsed time for encoding {(elapsed / 60):5.2f} min")

        logger.info(f"Saving corpus to '{cached_file}'")
        torch.save(corpus, cached_file)

    corpus.tokenizer = tokenizer
    assert corpus.use_moses == (args.mode == "moses")

    return corpus


def add_cls_token_to_tokenizer(tokenizer, log_name, cls_token='<cls>', add_idx=None):
    logger = logging.getLogger(log_name)
    logger.info(f"Tokenizer length: {len(tokenizer)}")

    # Add CLS token
    num = tokenizer.add_special_tokens({'cls_token': cls_token})
    logger.info(f"Added token '{cls_token}'")
    logger.info(f"New tokenizer length: {len(tokenizer)}")

    # Move CLS token to first embedding layer
    if add_idx is not None:
        old_cls_id = tokenizer.cls_token_id
        tokenizer.move_added_token(tokenizer.cls_token, add_idx)
        logger.info(f"Moved '{cls_token}' from position {old_cls_id} to {tokenizer.cls_token_id}")

    logger.info("=" * 100)

    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    utils.add_corpus_args(parser)
    utils.add_devices_args(parser)
    parser.add_argument('--mode', default='moses', choices=['base', 'move', 'copy', 'moses'])

    args = parser.parse_args()
    utils.prepare_devices(args)
    utils.create_logger(args)

    build_corpus(args)
