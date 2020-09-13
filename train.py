import argparse
import gc
import itertools
import logging
import math
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from apex import amp
from torch.nn.parallel import DistributedDataParallel, DataParallel
from tqdm import tqdm
from transformers import TransfoXLLMHeadModel
from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import AdamW

from modules import utils
from modules.data_builder import build_corpus
from modules.data_utils import LMOrderedIteratorHuggFace, Splits
from test_model import generate_summaries, calc_rouge_new


def show_mem_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def parse_args():
    parser = argparse.ArgumentParser()

    utils.add_devices_args(parser)      # Devices
    utils.add_training_args(parser)     # Training
    utils.add_eval_args(parser)         # Eval
    utils.add_generation_args(parser)   # Text generation settings
    utils.add_corpus_args(parser)       # Corpus settings

    args = parser.parse_args()
    if args.eval_bptt > 0:
        assert args.bptt == args.eval_bptt
    else:
        args.eval_bptt = args.bptt
    assert os.path.exists(args.data_dir)

    if not args.debug:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
    assert args.debug or os.path.exists(args.save_dir), "Argument '--save_dir' is required when not in debug mode"

    if args.loss_over_tgt_only:
        args.eval_loss_over_tgt_only = True

    if args.mode == "copy" or args.mode == "moses":
        args.copy_emb_weights = True

    if args.eval_only and args.train_name == "":
        args.train_name = "eval"

    if args.generate_only:
        args.eval_only = True

    utils.prepare_devices(args)
    if args.distributed:
        assert args.multi_gpu, "--distributed is only possible with --multi_gpu"

    return args


def mask_source_tokens(target_ids, summary_mask):
    """Sets targets of source tokens to -100 so that loss is only computed over summary tokens"""
    src_indices = (summary_mask.view(-1) <= 0).nonzero().squeeze()
    tgt = target_ids.view(-1)
    tgt[src_indices] = -100
    tgt_numel = tgt.size(0) - src_indices.size(0)
    return tgt.view(target_ids.size()), tgt_numel


def copy_embedding_weights(model, layer, copy_idx, target_idx):
    embed = model.get_input_embeddings()
    embed.emb_layers[layer].weight.data[target_idx, :] = embed.emb_layers[layer].weight.data[copy_idx, :].clone()
    model.tie_weights()


# coding: utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def train(args, train_iter, val_iter, model: TransfoXLLMHeadModel, para_model, optimizer, scheduler, epoch,
          train_step, train_summary, board_writer: utils.TensorboardWrapper, best_val_loss):
    logger = logging.getLogger(args.logger_name)
    model.train()

    train_loss = 0
    target_tokens = 0
    log_step = 0
    log_start_time = time.time()
    log_interval = 10
    skipped = 0
    estimate_train_time = True
    avg_elapsed_s = []

    mems = [None for _ in range(args.batch_chunk)]
    # logfile = open(os.path.join(args.save_dir, "log_batches.txt"), 'w')

    for batch, (data, target, seq_len, summary_mask) in enumerate(train_iter):
        target = data.clone()       # hotfix for new version of `transformers` where target is shifted inside the model
        log_step += 1
        target_tokens += target.numel()

        # logfile.write(f"Batch {batch}:\n{data}\n\n")

        if seq_len != args.bptt:
            skipped += 1
            logger.warning(f"Batch #{batch} has seq_len={seq_len} not matching tgt_len={args.bptt}. "
                           f"This batch is skipped!")
            logger.warning(f"# Skipped batches: {skipped}")
            continue

        model.zero_grad()
        curr_loss = 0



        # Chunk batch into mini-batches
        data_chunks = torch.chunk(data, args.batch_chunk)
        target_chunks = torch.chunk(target, args.batch_chunk)
        if args.loss_over_tgt_only:
            summary_mask_chunks = torch.chunk(summary_mask, args.batch_chunk)

        for i in range(args.batch_chunk):
            data_i = data_chunks[i].contiguous()
            target_i = target_chunks[i].contiguous()
            if args.loss_over_tgt_only:
                summary_mask_i = summary_mask_chunks[i].contiguous()
                target_i, target_numel = mask_source_tokens(target_i, summary_mask_i)

            outputs = para_model(input_ids=data_i, labels=target_i, mems=mems[i], return_tuple=True)
            loss, _, mems[i] = outputs[:3]

            if args.loss_over_tgt_only:
                if target_numel <= 0:
                    continue
                loss = loss.view(-1)[:target_numel]

            loss = loss.float().mean().type_as(loss) / args.batch_chunk
            if loss.item() == 0:
                continue

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.float().item()
            curr_loss += loss.float().item()

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # step-wise learning rate annealing
        train_step += 1
        optimizer.step()
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        train_summary['loss'].append(curr_loss)
        train_summary['lr'].append(lr)

        if train_step % log_interval == 0:
            mean_loss = train_loss / log_step       # mean loss over n steps (n = log_interval)
            mean_loss = utils.dist_all_reduce_item(mean_loss, 'mean')

            train_loss = 0
            if not args.debug and args.local_rank == 0:
                board_writer.w().add_scalar('train_loss', mean_loss, train_step)

            elapsed = time.time() - log_start_time
            avg_elapsed = elapsed / log_step
            avg_elapsed = utils.dist_all_reduce_item(avg_elapsed, 'max')
            log_start_time = time.time()
            log_step = 0
            if estimate_train_time:
                avg_elapsed_s.append(avg_elapsed)

            throughput = target_tokens / elapsed
            throughput = utils.dist_all_reduce_item(throughput, 'sum')
            target_tokens = 0

            log_str = '| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} ' \
                '| ms/batch {:5.1f} | tok/s {:7.0f} | loss {:5.2f}'.format(
                    epoch,
                    train_step,
                    batch+1,
                    train_iter.n_batch,
                    lr,
                    avg_elapsed * 1000,
                    throughput,
                    mean_loss
                    )

            logger.info(log_str)

        if train_step % args.eval_interval == 0:
            # Estimate training time
            if estimate_train_time:
                avg_time_per_batch = sum(avg_elapsed_s[1:]) / (len(avg_elapsed_s) - 1)
                avg_train_time = avg_time_per_batch * (args.max_steps - batch)
                # train time for whole data is ~50min
                eval_int = args.eval_interval if not args.debug else 20_000
                total_eval_time = (args.max_steps // eval_int) * (50*60)
                avg_train_time += total_eval_time
                logger.info('-' * 100)
                logger.info(f"Estimated time for training on whole data: {timedelta(0, avg_train_time)}")
                estimate_train_time = False

            logger.info('-' * 100)
            logger.info('Starting evaluation on validation data.')

            # Run on validation data.
            val_start_time = time.time()
            val_loss = evaluate(val_iter, model, args)
            val_loss = utils.dist_all_reduce_item(val_loss, 'mean')

            # Predict on validation data.
            val_acc = evaluate_predictions(val_iter, model, args)
            val_acc = utils.dist_all_reduce_item(val_acc, 'mean')
            val_elapsed = time.time() - val_start_time
            train_summary['valid_loss'].append(val_loss)
            if not args.debug and args.local_rank == 0:
                board_writer.w().add_scalar('valid_loss', val_loss, train_step)
                board_writer.w().add_scalar('valid_accuracy', val_acc, train_step)

            log_str = '| End of validation | validation time: {:5.2f}s | validation loss {:5.2f} ' \
                      '| validation ppl {:9.3f} | validation accuracy {:5.2f}'.format(
                    val_elapsed, val_loss, math.exp(val_loss), val_acc
                )
            logger.info('=' * len(log_str))
            logger.info(log_str)
            logger.info('=' * len(log_str))

            # Save checkpoint if validation loss is the best so far
            if (not best_val_loss or val_loss < best_val_loss) and not args.debug and args.local_rank == 0:
                best_val_loss = val_loss
                path = os.path.join(args.save_dir, 'model')
                if not os.path.exists(path):
                    os.mkdir(path)
                logger.info(f'Saving checkpoint to {path}')
                model.save_pretrained(path)

                path = os.path.join(args.save_dir, 'train_summary.pt')
                torch.save(train_summary, path)

        if train_step % 500 == 0 and args.local_rank == 0:
            utils.plot_curve(train_summary, args, "plots")
            logger.info(f"Saved loss curve plot in {args.save_dir}")

        if train_step == args.max_steps:
            break

    # logfile.close()
    return train_step, train_summary, best_val_loss


def evaluate(eval_iter, model, args):
    torch.cuda.empty_cache()
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # model.reset_length(args.bptt, model.config.ext_len, mem_len=args.eval_mem_len)

    # Evaluation
    skipped = 0
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = None
        for i, (data, target, seq_len, summary_mask) in tqdm(enumerate(eval_iter),
                                                             total=eval_iter.n_batch,
                                                             desc='Evaluating on eval data',
                                                             unit=' batch',
                                                             mininterval=30):
            target = data.clone()       # hotfix for new version of `transformers` where target is shifted inside the model
            if seq_len != args.bptt:
                skipped += 1
                # tqdm.write(f"WARNING - Batch #{i} has seq_len={seq_len} not matching tgt_len={args.bptt.tgt_len}. "
                #            f"This batch is skipped!")
                continue

            if args.eval_loss_over_tgt_only:
                target, target_numel = mask_source_tokens(target, summary_mask)

            outputs = model(input_ids=data, labels=target, mems=mems)
            loss, _, mems = outputs[:3]

            if args.eval_loss_over_tgt_only:
                if target_numel <= 0:
                    continue
                loss = loss.view(-1)[:target_numel]
            loss = loss.float().mean()

            # assert (not mems) or all([m.size(0) == args.eval_mem_len for m in mems])
            total_loss += seq_len * loss.item()
            total_len += seq_len

    # Switch back to the training mode
    model.train()
    # model.reset_length(args.bptt, model.config.ext_len, mem_len=args.mem_len)

    return total_loss / total_len


def evaluate_predictions(eval_iter, model, args):
    torch.cuda.empty_cache()
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_len, total_correct = 0, 0
    with torch.no_grad():
        # Predictions over whole vocabulary require more memory, thus data is chunked
        assert eval_iter.bsz % 4 == 0
        batch_chunk = eval_iter.bsz // 4    # chunk such that the chunk-batch-size gets 4
        mems = [None for _ in range(batch_chunk)]
        for idx, (data, target, seq_len, summary_mask) in tqdm(enumerate(eval_iter),
                                                               total=eval_iter.n_batch,
                                                               desc='Predicting on eval data',
                                                               unit=' batch',
                                                               mininterval=30):
            if seq_len != args.bptt:
                continue

            if args.eval_loss_over_tgt_only:
                target, target_numel = mask_source_tokens(target, summary_mask)

            # Chunk batch into mini-batches
            data_chunks = torch.chunk(data, batch_chunk)
            target_chunks = torch.chunk(target.cpu(), batch_chunk)
            correct_preds = 0

            for i in range(batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                outputs = model(input_ids=data_i, mems=mems[i])
                preds, mems[i] = outputs[:2]

                pred_max = torch.argmax(preds, dim=2).cpu()
                assert pred_max.size() == target_i.size()

                if args.eval_loss_over_tgt_only:
                    # Compare target summary tokens with predicted ones
                    if target_numel <= 0:
                        continue
                    tgt_idxs = (target_i.view(-1) >= 0).nonzero().squeeze()
                    tgt_pred_max = pred_max.view(-1)[tgt_idxs]
                    tgt_target = target_i.view(-1)[tgt_idxs]
                    correct_preds += len((tgt_pred_max == tgt_target).nonzero())
                else:
                    # Compare all targets tokens with predicted ones
                    correct_preds += len((pred_max == target_i).nonzero())

            total_correct += correct_preds
            if args.eval_loss_over_tgt_only:
                total_len += target_numel
            else:
                total_len += seq_len * eval_iter.bsz

    # Switch back to the training mode
    model.train()

    return total_correct / total_len


def main():
    args = parse_args()
    utils.create_logger(args)
    logger = logging.getLogger(args.logger_name)
    logger.info(f"Passed args: {' '.join(sys.argv)}")
    for key, value in vars(args).items():
        if value is not None:
            logger.info(f"{key: <25}{value}")

    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device, args.local_rank)
    if args.distributed:
        dist.init_process_group("nccl", rank=args.local_rank, world_size=args.world_size,
                                init_method='env://')

    # Init tensorbard
    board_writer = None
    if not args.debug and args.local_rank == 0:
        bsname = os.path.basename(args.save_dir)
        if bsname == '':
            bsname = os.path.basename(args.save_dir[:-1])
        board_path = os.path.join(args.tb_log_dir, bsname, args.train_name)
        board_writer = utils.TensorboardWrapper(board_path)

    ###########################################################################
    # Load data
    ###########################################################################
    corpus = build_corpus(args)
    if args.truncate_examples:
        logger.info(f"Truncating examples: sources to {args.max_src_len}, targets to {args.max_tgt_len}")
        corpus.truncate_all_examples(args.max_src_len, args.max_tgt_len)

    args.corpus = corpus            # for debugging purposes

    n_gpus = torch.cuda.device_count()
    if args.multi_gpu and not args.distributed:
        global_batch_size = args.batch_size * n_gpus
    else:
        global_batch_size = args.batch_size

    train_iter = LMOrderedIteratorHuggFace(corpus.get_data_flat(Splits.TRAIN), global_batch_size, args.bptt,
                                           summary_mask=corpus.get_summary_mask_flat(Splits.TRAIN), device=device)

    # wt-103
    # wt_103_path = 'prepro_temp/test.txt'
    # assert os.path.exists(wt_103_path), "Test data path does not exist"
    # corpus.data[Splits.TEST] = corpus.tokenizer.encode_file(wt_103_path, ordered=True)
    test_iter = LMOrderedIteratorHuggFace(corpus.get_data_flat(Splits.TEST), args.eval_batch_size, args.eval_bptt,
                                          summary_mask=corpus.get_summary_mask_flat(Splits.TEST), device=device)
    val_iter = LMOrderedIteratorHuggFace(corpus.get_data_flat(Splits.VALID), args.eval_batch_size, args.eval_bptt,
                                         summary_mask=corpus.get_summary_mask_flat(Splits.VALID), device=device)

    train_summary = {'loss': [], 'lr': [], 'valid_loss': [], 'test_loss': -1.0, 'epochs': []}

    if args.max_epochs > 0:
        args.max_steps = train_iter.n_batch * args.max_epochs

    ###########################################################################
    # Load Model
    ###########################################################################
    # resized_model_path = os.path.join(args.data_dir, 'resized_model')
    if not args.eval_only:
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103',
                                                     dropout=args.dropout,
                                                     dropatt=args.dropout_att)
        if model.config.vocab_size != len(corpus.tokenizer):
            logger.info(f"Model and tokenizer have not same length / vocab_size (model: {model.config.vocab_size}, "
                        f"tokenizer: {len(corpus.tokenizer)})")

            new_token_layer = 0 if args.mode != "base" else -1
            logger.info(f"Resizing the embedding layer {new_token_layer}")
            model.resize_token_embeddings(len(corpus.tokenizer), layer=new_token_layer)
            if args.copy_emb_weights:
                logger.info("Copying embedding weights from <eos> token to <cls> token.")
                copy_embedding_weights(
                    model, new_token_layer, args.corpus.tokenizer.eos_token_id, args.corpus.tokenizer.cls_token_id
                )
        # model.save_pretrained(resized_model_path)
        #
        # logger.info(f"Loading model from {resized_model_path}.")
        # model = TransfoXLLMHeadModel.from_pretrained(resized_model_path,
        #                                              dropout=args.dropout,
        #                                              dropatt=args.dropout_att)

        assert corpus.tokenizer.cls_token == '<cls>'
        assert model.config.vocab_size == len(corpus.tokenizer)

        model.reset_length(args.bptt, model.config.ext_len, args.mem_len)
        model.config.tgt_len = args.bptt
        model.config.mem_len = args.mem_len
        model.to(device)

        n_all_param = sum([p.nelement() for p in model.parameters()])
        n_nonemb_param = sum([p.nelement() for p in model.transformer.layers.parameters()])
        logger.info(f"#Params: {n_all_param}")
        logger.info(f"#Non Emb Params: {n_nonemb_param}")

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warm_steps, args.max_steps)

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", max_loss_scale=2**16)

        if args.multi_gpu and not args.distributed:
            para_model = DataParallel(model).to(device)
            logger.info(f"Using {n_gpus} GPUs with a global batch size of {global_batch_size}!")
        elif args.distributed:
            para_model = DistributedDataParallel(
                model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True,
            )
        else:
            para_model = model

        ###########################################################################
        # Train
        ###########################################################################

        train_step = 0
        best_val_loss = None

        # Loop over epochs.
        # At any point you can hit Ctrl + C to break out of training early.
        epoch_str = f"{args.max_epochs} epochs aka " if args.max_epochs > 0 else ""
        logger.info(f"Starting training for {epoch_str}{args.max_steps} steps.")
        start_time = time.time()
        try:
            for epoch in itertools.count(start=1):
                train_step, train_summary, best_val_loss = train(
                    args, train_iter, val_iter, model, para_model, optimizer, scheduler, epoch,
                    train_step, train_summary, board_writer, best_val_loss
                )

                if train_step == args.max_steps or epoch == args.max_epochs:
                    logger.info('-' * 100)
                    logger.info('End of training')
                    break

                train_summary['epochs'].append(train_step)

        except KeyboardInterrupt:
            logger.info('-' * 100)
            logger.info('Exiting from training early')
            return

        elapsed = time.time() - start_time
        logger.info(f"Elapsed time: {timedelta(0, elapsed)}")

    ###########################################################################
    # Test
    ###########################################################################
    del train_iter
    del val_iter
    test_path = os.path.join(args.save_dir, 'model')
    if not os.path.exists(test_path):
        logger.error(f"Tried to load model for testing from {test_path}. Path does not exist!")
        return

    # Reload model checkpoint
    if not args.eval_only:
        del model
    torch.cuda.empty_cache()
    time.sleep(5)
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB")
    logger.info(f"GPU memory cached:    {torch.cuda.memory_cached(device) / (1024 ** 3):.2f} GB")
    model = TransfoXLLMHeadModel.from_pretrained(test_path, clamp_len=0)

    # Increase mem_len for evaluation and generation
    model.reset_length(args.eval_bptt, model.config.ext_len, mem_len=args.eval_mem_len)
    model.to(device)
    logger.info('-' * 100)

    if not args.generate_only:
        logger.info('Starting evaluation on test data.')
        # Run on test data.
        test_start_time = time.time()
        test_loss = evaluate(test_iter, model, args)
        # Predict on test data.
        test_acc = evaluate_predictions(test_iter, model, args)
        test_elapsed = time.time() - test_start_time
        train_summary['test_loss'] = test_loss

        logger.info('=' * 100)
        logger.info(
            '| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f} | test accuracy {:5.2f}'.format(
                test_elapsed, test_loss, math.exp(test_loss), test_acc))
        logger.info('=' * 100)

    # Generate summaries
    logger.info("Starting with generation of summaries from test data.")
    test_start_time = time.time()
    generated = generate_summaries(args, model, corpus, device, Splits.TEST)
    test_elapsed = time.time() - test_start_time
    logger.info("Time for generation: {:.2f} min".format((test_elapsed / 60)))
    rouge_scores = calc_rouge_new(args, generated, Splits.TEST)
    # rouge_scores = get_dummy_rouge_scores()

    metrics = {} if args.generate_only else {'test_loss': test_loss, 'test_accuracy': test_acc}
    for i, k1 in enumerate(rouge_scores.keys()):
        metrics[k1] = rouge_scores[k1]['f']
    if not args.debug:
        delattr(args, "corpus")  # Remove corpus before writing it to tensorboard
        utils.metrics2hparams(metrics)
        tb_args = utils.without_args(vars(args), utils.TENSORBOARD_IGNORE_ARGS)
        board_writer.w().add_hparams(hparam_dict=tb_args, metric_dict=metrics)

    # Save statistics
    if not args.eval_only:
        path = os.path.join(args.save_dir, f'train_summary.pt')
        torch.save(train_summary, path)

    return


if __name__ == "__main__":
    main()
