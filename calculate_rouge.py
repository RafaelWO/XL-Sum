import argparse
import os
import pickle
import shutil
import time

import nltk
from pyrouge import Rouge155
from rouge import Rouge

import modules.data_utils as data_utils
from modules.ExtendedMosesTokenizer import ExtendedMosesTokenizer


def prepare_for_rouge(texts, split_sentences=True, tokenize=True, only_first_sentence=False):
    moses_tok = ExtendedMosesTokenizer()
    prepared_texts = []

    for text in texts:
        text = text.replace("<unk>", "UNK")
        text = text.replace("<eos>", "")

        if tokenize:
            text = moses_tok.tokenize(text)
            text = " ".join(text)

        if split_sentences:
            sents = nltk.sent_tokenize(text)
            if only_first_sentence:
                text = sents[0]
            else:
                text = "\n".join(sents)

        prepared_texts.append(text)
    return prepared_texts


def write_rouge_files(source_file, dest_file, only_first_sentence=False):
    texts = data_utils.read_generated_summaries(source_file)
    texts = prepare_for_rouge(texts, only_first_sentence=only_first_sentence)

    for counter, line in enumerate(texts):
        with open(f"{dest_file}.{str(counter + 1).zfill(5)}.txt", "w") as file:
            file.write(line)

    print(f"Done writing {counter + 1} files from {source_file}.")


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', required=True)
    parser.add_argument('--post', default="eval")
    parser.add_argument('--against_plain', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--single', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--first_sent', type=bool, default=False, nargs='?', const=True)

    args = parser.parse_args()

    print("Calculating ROUGE scores for the following experiments:", ", ".join(args.exp))
    time.sleep(3)
    for exp in args.exp:
        # Set up path
        path_to_exp = os.path.join("trainings", exp)
        assert os.path.exists(path_to_exp)

        generated_path = "rouge/system_summaries"  # Generated
        gold_path = "rouge/model_summaries"  # Gold

        hyps = []
        refs = []

        ### Generated
        if not args.single:
            # Clean rouge dirs
            for folder_path in [generated_path, gold_path]:
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                os.mkdir(folder_path)

            write_rouge_files(os.path.join(path_to_exp, f"generated_summaries_{args.post}.txt"),
                              os.path.join(generated_path, "tmp"), args.first_sent)
        else:
            hyps = data_utils.read_generated_summaries(
                os.path.join(path_to_exp, f"generated_summaries_{args.post}.txt"))
            hyps = prepare_for_rouge(hyps, only_first_sentence=args.first_sent)

        path_to_tgt = os.path.join(path_to_exp, "target_summaries_new.txt")
        if args.against_plain:
            path_to_tgt = os.path.join("rouge", "target_summaries_plain.txt")
            print("Comparing against plain text gold summaries ")

        ### Targets
        if not args.single:
            write_rouge_files(path_to_tgt, os.path.join(gold_path, "tmp"), args.first_sent)
        else:
            refs = data_utils.read_generated_summaries(path_to_tgt)
            refs = prepare_for_rouge(refs, only_first_sentence=args.first_sent)

        postfix = ""
        if args.post is not "":
            postfix = "_" + args.post
        if args.against_plain:
            postfix += "_against-plain"
        if args.single:
            postfix += "_single"
        if args.first_sent:
            postfix += "_first-sent"

        if not args.single:
            print("Executing pyrouge...")
            time.sleep(5)
            r = Rouge155()
            r.system_dir = generated_path
            r.model_dir = gold_path
            r.system_filename_pattern = 'tmp.(\d+).txt'
            r.model_filename_pattern = 'tmp.#ID#.txt'

            output = r.convert_and_evaluate()
            print(output)
            postfix = ""
            if args.post is not "":
                postfix = "_" + args.post
            if args.against_plain:
                postfix += "_against-plain"
            with open(f"rouge/results_{exp}{postfix}.txt", "w") as file:
                file.write(output)

        else:
            print("Executing rouge...")
            rouge = Rouge()
            scores = rouge.get_scores(hyps, refs, avg=False)
            with open(f"rouge/results_{exp}{postfix}.pkl", "wb") as file:
                pickle.dump(scores, file)
