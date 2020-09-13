import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from modules.CustomSummaryWriter import CustomSummaryWriter

TENSORBOARD_IGNORE_ARGS = {
    "device",
    "world_size"
    "multi_gpu",
    "local_rank",
    "save_dir",
    "debug",
    "train_name",
    "eval_only",
    "eval_raw",
    "eval_interval",
    "data_dir",
    "logger_name",
    "rebuild_corpus",
    "distributed",
    "copy_emb_weights"
    "tb_log_dir"
}


def add_devices_args(parser):
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu', default='6',
                        help="GPU ids separated by a comma (no space!)")
    parser.add_argument('--multi_gpu', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--distributed', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--local_rank', type=int,
                        default=os.getenv('LOCAL_RANK', 0),
                        help='Used for multi-process training.')
    parser.add_argument('--world_size', type=int,
                        default=os.getenv('WORLD_SIZE', 1))


def add_training_args(parser):
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Defines the (local) batch size. If multiple GPUs are used, '
                             'this batch size is used per one GPU, i.e. the global batch '
                             'size = #gpus * batch_size')
    parser.add_argument('--bptt', type=int, default=192)
    parser.add_argument('--mem_len', type=int, default=192)
    parser.add_argument('--warm_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss_over_tgt_only', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_att', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--adam_epsilon', type=float, default=1e-9)
    parser.add_argument('--batch_chunk', type=int, default=2,
                        help='Split batch into chunks and train with gradient accumulation')
    parser.add_argument('--copy_emb_weights', type=bool, default=False, nargs='?', const=True)

    parser.add_argument('--save_dir')
    parser.add_argument('--debug', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--train_name', default="")
    parser.add_argument('--fp16', type=bool, default=False, nargs='?', const=True,
                        help='Run training in fp16/mixed precision')
    parser.add_argument('--tb_log_dir', default='tensorboard_final')


def add_eval_args(parser):
    parser.add_argument('--eval_only', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_bptt', type=int, default=0,
                        help="If this is 0 it will be set to the same value as '--bptt'.")
    parser.add_argument('--eval_mem_len', type=int, default=800)
    parser.add_argument('--eval_raw', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--eval_loss_over_tgt_only', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--eval_interval', type=int, default=5000)


def add_generation_args(parser):
    parser.add_argument('--generate_only', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--min_summary_len', type=int, default=40)
    parser.add_argument('--max_summary_len', type=int, default=75)
    parser.add_argument('--repetition_penalty', type=float, default=2.0)
    parser.add_argument('--summary_max_src_len', type=int, default=1500)
    parser.add_argument('--beams', type=int, default=1)
    parser.add_argument('--len_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram', type=int, default=3)
    parser.add_argument('--do_sample', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--temp', type=float, default=1.0)


def add_corpus_args(parser):
    parser.add_argument('--data_dir', default='prepro_temp')
    parser.add_argument('--logger_name', default='Main')
    parser.add_argument('--rebuild_corpus', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--corpus_pct', default='100')
    parser.add_argument('--truncate_examples', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--max_src_len', type=int, default=750)  # default from t2t cnn_dm summarization
    parser.add_argument('--max_tgt_len', type=int, default=100)  # default from t2t cnn_dm summarization
    parser.add_argument('--mode', default='moses', choices=['base', 'move', 'copy', 'moses'])


def prepare_devices(args):
    gpus = args.gpu.split(",")
    gpus = list(filter(None, gpus))  # Remove empty entries
    args.multi_gpu = True if len(gpus) > 1 else False
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)


def without_args(d: dict, ignore_keys):
    return {x: d[x] for x in d if x not in ignore_keys}


def create_logger(args):
    # create logger
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.INFO)

    # create formatter
    if getattr(args, "distributed", False):
        formatter = logging.Formatter('%(asctime)s %(name)s-%(levelname)s|%(rank)s: %(message)s',
                                  datefmt='%d.%m.%Y %H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s %(name)s-%(levelname)s: %(message)s',
                                      datefmt='%d.%m.%Y %H:%M:%S')
    # console logger
    # if args.debug:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    if getattr(args, "distributed", False):
        rank_filter = RankFilter(args.local_rank, True)
        logger.addFilter(rank_filter)


def plot_curve(summary, args, fname):
    steps = len(summary['loss'])
    ticks = steps // 20
    valid_idx = np.arange(args.eval_interval, steps + 1, args.eval_interval)

    plt.figure(figsize=(28, 16))
    plt.subplot(211)
    # plot loss
    plt.plot(summary['loss'], label='loss')
    # plot valid loss
    if len(summary['valid_loss']) > 0:
        plt.plot(valid_idx, summary['valid_loss'], label='validation loss',
                 color='yellow', marker='o', markersize=10)
        for x, y in zip(valid_idx, summary['valid_loss']):
            plt.annotate("%.2f" % y,
                         xy=(x, y), xycoords='data',
                         xytext=(15, 10), textcoords='offset points',
                         # arrowprops=dict(arrowstyle='->'),#facecolor='black', shrink=0.05),
                         horizontalalignment='left', verticalalignment='bottom')
    # plot test loss
    if summary['test_loss'] > -1:
        plt.plot(steps, summary['test_loss'], label='test loss',
                 color='red', marker='*', markersize=10)
        plt.annotate("%.2f" % summary['test_loss'],
                     xy=(steps, summary['test_loss']), xycoords='data',
                     xytext=(30, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'),  # facecolor='black', shrink=0.05),
                     horizontalalignment='left', verticalalignment='bottom')

    if len(summary['epochs']) > 0:
        plt.vlines(summary['epochs'], ymin=np.min(summary['loss']), ymax=np.max(summary['loss']),
                   label='epoch border', linestyles='dashed')

    plt.xticks(np.arange(0, steps + 1, ticks))
    plt.title('Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(summary['lr'])
    plt.xticks(np.arange(0, len(summary['loss']) + 1, ticks))
    plt.title('Learning Rate Curve')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')

    path = os.path.join(args.save_dir, fname + '.png')
    plt.savefig(path)
    plt.close()


def metrics2hparams(metrics_dict: dict):
    keys = list(metrics_dict.keys())
    for k in keys:
        if k.startswith("hparam/"):
            continue
        new_k = f"hparam/{k}"
        metrics_dict[new_k] = metrics_dict.pop(k)


class RankFilter(logging.Filter):
    def __init__(self, rank, log_all_ranks):
        super().__init__()
        self.rank = rank
        self.log_all_ranks = log_all_ranks

    def filter(self, record):
        record.rank = self.rank
        if self.log_all_ranks:
            return True
        else:
            return (self.rank == 0)


class TensorboardWrapper(object):
    def __init__(self, path):
        self.path = path
        self.writer = None

    def w(self):
        if self.writer is None:
            self.writer = CustomSummaryWriter(self.path)

        return self.writer


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
def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    return world_size


def dist_all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if dist.is_available() and dist.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = dist.ReduceOp.SUM
        elif op == 'min':
            dop = dist.ReduceOp.MIN
        elif op == 'max':
            dop = dist.ReduceOp.MAX
        elif op == 'product':
            dop = dist.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')

        backend = dist.get_backend()
        if backend == dist.Backend.NCCL:
            device = torch.device('cuda')
        elif backend == dist.Backend.GLOO:
            device = torch.device('cpu')
        else:
            raise RuntimeError('Unsupported distributed backend')

        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, dop)
        if op == 'mean':
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret