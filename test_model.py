import logging
import os

import numpy as np
import torch
from pyrouge import Rouge155
from rouge import Rouge
from tqdm import tqdm

import modules.data_utils as data_utils
from calculate_rouge import prepare_for_rouge
from modules import utils
from modules.data_builder import build_corpus


def generate_summaries(args, model, corpus, device, split=data_utils.Splits.TEST):
    logger = logging.getLogger(args.logger_name)
    model.eval()

    cls_tok = corpus.tokenizer.cls_token_id
    generated_texts = []
    truncated = 0
    for i, sample in tqdm(enumerate(corpus.data[split]),
                          desc='Generating summaries',
                          unit=' examples',
                          total=len(corpus.data[split]),
                          mininterval=(5*60)):
        #if i % 5 == 0:
        torch.cuda.empty_cache()

        src, _ = data_utils.split_example(sample, cls_tok)
        if src.numel() > args.summary_max_src_len:
            truncated += 1

        try:
            ids, _ = generate_summary(
                src, model, corpus.tokenizer, args.min_summary_len, args.max_summary_len, device,
                args.summary_max_src_len, args.repetition_penalty, args.beams, args.len_penalty, args.no_repeat_ngram,
                args.do_sample, args.temp
            )
        except RuntimeError as e:
            logger.error(f"\nRuntimeError at sample No. {i}. Exiting...")
            raise e

        text = args.corpus.decode_ids(ids)
        generated_texts.append(text)

    logger.info("Done generating summaries")
    logger.info(f"{truncated}/{len(corpus.data[split])} samples were truncated to {args.summary_max_src_len}")

    summary_file = os.path.join(args.save_dir, f'generated_summaries_{args.train_name}.txt')
    with open(summary_file, 'w', encoding='utf-8') as file:
        for idx, text in enumerate(generated_texts):
            file.write(f"{idx}:\t {text}\n\n")

    model.train()
    return generated_texts


def generate_summary(text, model, tokenizer, min_out_len, max_out_len, device, max_in_len=1500, repetition_penalty=2.0,
                     beams=1, length_penalty=1.0, no_repeat_ngram_size=0, do_sample=False, temperature=1.0,
                     priming=None,
                     append_cls=True):
    if type(text) == str:
        source_tok = tokenizer.encode(text, add_space_before_punct_symbol=True)
        src = torch.tensor(source_tok)
    elif type(text) == torch.Tensor:
        src = text
    else:
        raise TypeError("Input must be either a string or a tensor")

    src_len = src.numel()
    # print(f"Percentage of unknown tokens: {((source_tok.count(tokenizer.unk_token_id) / input_len) * 100):.2f}%")
    if src_len > max_in_len:
        src = src[:max_in_len]

    if append_cls:
        src = torch.cat((src, torch.tensor([tokenizer.cls_token_id])))
    if priming:
        src = torch.cat((src, torch.tensor(tokenizer.encode(priming, add_space_before_punct_symbol=True))))

    input_len = src.numel()

    in_ids = src.unsqueeze(0).to(device)
    output = model.generate(in_ids, pad_token_id=0,
                            do_sample=do_sample,
                            num_beams=beams,
                            min_length=input_len + min_out_len,
                            max_length=input_len + max_out_len,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            early_stopping=True,
                            temperature=temperature)
    summary = output[0][input_len:]
    summary.cpu()
    src.cpu()
    return summary, src


def calc_rouge(args, generated_texts, split=data_utils.Splits.TEST):
    cnndm = data_utils.get_cnndm_dataset(split, args.corpus_pct)

    hyps = []
    refs = []
    target_file = os.path.join(args.save_dir, 'target_summaries.txt')
    file = open(target_file, 'w')
    for idx, sample in tqdm(enumerate(cnndm), desc='Writing articles', unit=' examples'):
        src, tgt = sample['article'], sample['highlights']
        # src, tgt = tf.strings.as_string(src), tf.strings.as_string(tgt)
        src, tgt = bytes.decode(src.numpy()), bytes.decode(tgt.numpy())
        # src = bytes.decode(src.numpy(), errors='replace', encoding='utf-8')
        # tgt = bytes.decode(tgt.numpy(), errors='replace', encoding='utf-8')
        tgt = tgt.replace("\n", " ")
        pred = generated_texts[idx]
        file.write(f"{idx}:\t {tgt}\n\n")

        refs.append(tgt)
        hyps.append(pred)

    file.close()
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    print_rouge_result(args, scores)
    return scores


def calc_rouge_new(args, generated_texts=None, split=data_utils.Splits.TEST, calc_len=True):
    logger = logging.getLogger(args.logger_name)
    if generated_texts is None:
        source_file = os.path.join(args.save_dir, f'generated_summaries_{args.train_name}.txt')
        generated_texts = data_utils.read_generated_summaries(source_file)

    data = args.corpus.data[split]
    refs = []
    summary_lens = []

    target_file = os.path.join(args.save_dir, 'target_summaries_new.txt')
    file = open(target_file, 'w')
    for idx, sample in tqdm(enumerate(data), desc='Processing targets', unit=' examples'):
        _, tgt_ids = data_utils.split_example(sample, args.corpus.tokenizer.cls_token_id)
        tgt = args.corpus.decode_ids(tgt_ids)
        file.write(f"{idx}:\t {tgt}\n\n")

        refs.append(tgt)

        if calc_len:
            pred_ids = args.corpus.encode_text(generated_texts[idx])
            summary_lens.append((len(pred_ids), len(tgt_ids)))
    file.close()

    generated_texts = prepare_for_rouge(generated_texts, True, True)
    refs = prepare_for_rouge(refs, True, True)
    assert len(generated_texts) == len(refs)

    rouge = Rouge()
    scores = rouge.get_scores(generated_texts, refs, avg=True)
    print_rouge_result(args, scores)
    if calc_len:
        mean_len = np.average(summary_lens, 0)
        logger.info(f"Mean generated summary length: {mean_len[0]:.2f}")
        logger.info(f"Mean target summary length: {mean_len[1]:.2f}")
    return scores


def calc_rouge_files_test(args, generated_texts):
    logger = logging.getLogger(args.logger_name)
    # cnndm_test = get_cnndm_dataset(split, args.corpus_pct)

    sys_sum_path = os.path.join(args.data_dir, 'system_summaries')
    mod_sum_path = os.path.join(args.save_dir, 'model_summaries')
    if not os.path.exists(sys_sum_path):
        os.mkdir(sys_sum_path)
    if len(os.listdir(sys_sum_path)) == 0:
        data_utils.write_system_summaries(args.data_dir, args.corpus_pct)
    assert len(os.listdir(sys_sum_path)) == len(generated_texts)
    if not os.path.exists(mod_sum_path):
        os.mkdir(mod_sum_path)

    # Write summaries into files
    for idx, sample in enumerate(generated_texts):
        filepath = os.path.join(mod_sum_path, f'summary.{str(idx+1).zfill(4)}.txt')
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(sample)

    r = Rouge155()
    r.system_dir = sys_sum_path
    r.model_dir = mod_sum_path
    r.system_filename_pattern = 'summary.(\d+).txt'
    r.model_filename_pattern = 'summary.#ID#.txt'

    output = r.convert_and_evaluate()
    logger.info(output)

    return r.output_to_dict(output)


def print_rouge_result(args, scores):
    logger = logging.getLogger(args.logger_name)

    for i, k1 in enumerate(scores.keys()):
        line = "| {:s}: F: {:.4f} | P: {:.4f} | R: {:.4f} |".format(
            k1, scores[k1]['f'], scores[k1]['p'], scores[k1]['r'])

        if i == 0:
            logger.info("=" * len(line))

        logger.info(line)

    logger.info("=" * len(line))


def get_dummy_rouge_scores():
    return {'rouge-1': {'f': 0.0726116192706683, 'p': 0.10229403379604642, 'r': 0.05913724290847152},
            'rouge-2': {'f': 0.003909433129675029, 'p': 0.005644816666683782, 'r': 0.0031334451176694028},
            'rouge-l': {'f': 0.0681234699980458, 'p': 0.09064880641443637, 'r': 0.05689435274496617}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='prepro_temp')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--logger_name', default='Main')
    parser.add_argument('--rebuild_corpus', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--corpus_pct', default='5')
    parser.add_argument('--mode', default='moses', choices=['base', 'move', 'copy', 'moses'])
    # parser.add_argument('--gpu', default='6')
    parser.add_argument('--train_name', default='eval')
    # parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    utils.create_logger(args)

    args.corpus = build_corpus(args)
    calc_rouge_new(args, calc_len=False)
