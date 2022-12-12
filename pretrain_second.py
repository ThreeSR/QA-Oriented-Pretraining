# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import math
import os
import random
import sys
from io import open
from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import vilbert.utils as utils
from vilbert.task_utils import LoadSecondPretrainDatasets
# from vilbert.task_utils2 import LoadDatasets
from vilbert.vilbert import BertConfig, BertForMultiModalPreTraining
from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters

    # parser.add_argument(
    #     "--file_path",
    #     default="/dvmm-filer3a/users/rui/multi-task/datasets/VQA/cache",
    #     type=str,
    #     help="The input train corpus.",
    # )
    parser.add_argument( # 这边需要修改，因为这里是second pretrain，所以应该是从CC的pretrain权重进行读取
        "--from_pretrained",
        default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/pretrain/pretrained_model2.bin",
        #default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/pretrain/pytorch_model_1.bin",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-base-uncased, roberta-base, roberta-large, ",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/config/bert_base_6layer_6conect.json",
        help="The config file which specified the model details.",
    )
    ## Other parameters
    # parser.add_argument( # * 这个指令没有用到
    #     "--max_seq_length",
    #     default=23, # 这个是根据VQA任务改的
    #     type=int,
    #     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #     "Sequences longer than this will be truncated, and sequences shorter \n"
    #     "than this will be padded.",
    # )
    parser.add_argument( # 这一个变量在后面被重新赋值为128了，128 for VQA
        "--train_batch_size",
        default=128, # 原本是512 for CC， 128 for VQA，但是显存不够，所以换成64加上2个gradient accumulation
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument( # NOTE 这里的学习率改过
        "--learning_rate",
        default=4e-5, # 原本是1e-4，但vqa，gqa，vgqa都是4e-5
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument( # ! NOTE 这里会影响下面的学习率
        "--num_train_epochs",
        default=20, # 原本是10.0，现在改成20.0，现在改成2
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--img_weight", default=1, type=float, help="weight for image loss"
    )
    parser.add_argument( # 默认false
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument( # ！！！！
        "--gradient_accumulation_steps",
        type=int,
        default=1, # 原本是1
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument( # 默认false
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument( # 默认false
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument( # 原本是25！
        "--num_workers",
        type=int,
        default=16, # *vqa是16
        help="Number of workers in the dataloader.",
    )
    parser.add_argument( # NOTE 保存的名字！  "VQA_SecondPretrain7"
        "--save_name",
        default="VQA_VGQA_GQA_SecondPretrain2",
        # default="GQA_SecondPretrain3",
        # default="VGQA_SecondPretrain3",
        type=str,
        help="save name for training."
    )
    parser.add_argument(
        "--baseline",
        action="store_true", #store_true 是指带触发action 时为真，不触发则为假， 即默认False
        help="Wheter to use the baseline model (single bert).",
    )
    parser.add_argument( # 这一步关系到后续的vilbert中哪些层freeze住
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument( # 默认false
        "--distributed",
        action="store_true",
        help="whether use chunck for parallel training.",
    )
    parser.add_argument( # 默认false
        "--without_coattention", action="store_true", help="whether pair loss."
    )
    parser.add_argument(  # TODO: 这里暂时取1，之后取完特征，再改成0
        "--visual_target",
        default=1,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )

    parser.add_argument( # TODO: 这个和上面的visual target不是一回事；暂时改成1，感觉比较make sense
        "--objective",
        default=1,
        type=int,
        help="which objective to use \
        0: with ICA loss, \
        1: with ICA loss, for the not aligned pair, no masking objective, \
        2: without ICA loss, do not sample negative pair.",
    )
    parser.add_argument( # 默认false NOTE
        "--sep",
        action="store_true",
        help="Whether to use sep token between question and answer",
    )
    parser.add_argument( # 默认false NOTE
        "--only_ans_mlm",
        action="store_true",
        help="Whether to only use MLM in answer text",
    )
    parser.add_argument( # 默认false NOTE 其实暂时没用到，因为需要自己手动更改，函数被dataloader调用
        "--iou",
        action="store_true",
        help="Whether to use iou threshold to mask more regions",
    )
    parser.add_argument(
        "--num_negative", default=255, type=int, help="num of negative to use"
    )

    parser.add_argument( # 继续train
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--tasks", default="1", type=str, help="1-2-3... training task separate by -"
    ) # 代表进行fine tune哪一个任务
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--vision_scratch",
        action="store_true",
        help="whether pre-trained the image or not.",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="warmup_linear",
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--train_iter_multiplier",
        default=1.0,
        type=float,
        help="multiplier for the multi-task training.",
    )
    parser.add_argument(
        "--train_iter_gap",
        default=4,
        type=int,
        help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop",
    )

    # * -------------------------------------------------------------------------- * #
    # * 下面是多卡训练的配置
    parser.add_argument( # 开启多卡，使用torch.distributed.launch，届时这个参数会被传进来...不用管它
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    # 每个节点的GPU数量，1到8个GPU
    parser.add_argument('--gpus', # NOTE 这个参数要记得设定，如果跑多卡的话
                        default=1,
                        type=int,
                        help='Number of gpus per node'
    )
    # 节点数量，一台机子
    # parser.add_argument('--nodes',
    #                     default=1,
    #                     type=int,
    #                     help="Number of nodes in the cluster"
    # )
    # * -------------------------------------------------------------------------- * #

    # args 实例化
    args = parser.parse_args()

    if args.local_rank != -1:
        os.environ['MASTER_ADDR'] = '127.0.0.1' # * 这个就是IP地址，ifconfig查看 127.0.0.1 or 128.59.9.218
        os.environ['MASTER_PORT'] = '8888' # * 端口号需要自己输入一个现在不存在的...
    # * parse部分结束 * #

    with open("vilbert_tasks2.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.fp16:
        from apex import amp

    if args.baseline: # 默认是false
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BertForMultiModalPreTraining
    else: # 默认使用vilbert
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import BertForMultiModalPreTraining # BertForMultiModalPreTraining重名，但一般用vilbert的

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split("-")): # 1-2-3...输入任务，假设这时候只有任务一，在进行fine tune
        task = "TASK" + task_id  # TASK1
        name = task_cfg[task]["name"] # VQA
        task_names.append(name)
        task_lr.append(task_cfg[task]["lr"]) # 0.00004 学习率

    base_lr = min(task_lr)
    # ...
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.save_name: # 默认是空null
        prefix = "-" + args.save_name
    else:
        prefix = ""
    # "/home/rui/code/vilbert-multi-task/vilbert-multi-task/config/bert_base_6layer_6conect.json"
    timeStamp = args.config_file.split("/")[1].split(".")[0] + prefix # bert_base_6layer_6conect-VQA_second_stage_pretrain
    savePath = os.path.join(args.output_dir, timeStamp) # save/bert_base_6layer_6conect

    bert_weight_name = json.load( # config/bert-base-uncased_weight_name.json
        open("config/" + args.bert_model + "_weight_name.json", "r") # NOTE: 这边被我改过！
    ) # 读取权重的名字 "bert-base-uncased"

    # ******************************************************************** #
    # 多卡训练
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        # device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # n_gpu = 1 # NOTE !这个除了有记录的作用，在后面的data parallel中也有用！！！ 所以这个值的设置要注意
        n_gpu = args.gpus # !这个除了有记录的作用，在后面的data parallel中也有用！！！
        torch.distributed.init_process_group(backend="nccl") # env形式的初始化，此外还有tcp形式,不需要增加额外参数
    # ******************************************************************** #

    # device = torch.device('cuda:0')
    # torch.cuda.set_device(0)

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        ) # 默认不进行分布式训练
    )

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath) # 创建新文件夹

    config = BertConfig.from_json_file(args.config_file) # config/bert_base_6layer_6conect.json

    if default_gpu:
        # save all the hidden parameters. 把配置文件的内容存下来
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

#    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps # ？？ 为什么不是相乘
    # num_train_optimization_steps = None

    # cache = 5000
    # if dist.is_available() and args.local_rank != -1: # 不过
    #     num_replicas = dist.get_world_size()
    #     args.train_batch_size = args.train_batch_size // num_replicas
    #     args.num_workers = args.num_workers // num_replicas
    #     cache = cache // num_replicas

    # cased区分大小写，不需要lower-case
    # uncased不区分大小写(这句话是错的，
    # 所谓不区分大小写实际上是不能区分大小写，
    # 因为词表只有小写,这就是我为什么那么久还搞混的原因…)，需要lower-case
    # tokenizer = BertTokenizer.from_pretrained( # bert-base-uncased
    #     args.bert_model, do_lower_case=args.do_lower_case # TRUE
    # )

    # 在下面这行代码之中，由于数据集默认是trainval的名字，所以实际上train和val两个数据集是一起读取了
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = LoadSecondPretrainDatasets(
        args, task_cfg, args.tasks.split("-")
    )

    logdir = os.path.join("logs", timeStamp) # logs/bert_base_6layer_6conect-VQA_second_stage_pretrain
    if default_gpu:
        tbLogger = utils.tbLogger(
            logdir, # logs/bert_base_6layer_6conect-VQA_second_stage_pretrain
            savePath, # save/bert_base_6layer_6conect
            task_names, # ["VQA", "GQA" ...]
            task_ids, # ["TASK1"]
            task_num_iters, # 这个就是多少个batch，就是要iter多少次
            args.gradient_accumulation_steps,
        )

    # 注意： task_batch_size task_num_iters 等变量是字典！！
    # args.train_batch_size = task_batch_size["TASK1"]

    # num_train_optimization_steps = int(
    #     len(task_datasets_train["TASK1"]) # 原本是 task_datasets_train.num_dataset
    #     / args.train_batch_size # 512 128
    #     / args.gradient_accumulation_steps # 1
    # ) * (args.num_train_epochs - args.start_epoch) #  num_train_epochs=10，start=0

    # num_train_optimization_steps = int(
    #     train_dataset.num_dataset
    #     / args.train_batch_size
    #     / args.gradient_accumulation_steps
    # ) * (args.num_train_epochs - args.start_epoch)

    # TODO: need to revise
    # task_names = ["VQA_Pretrain2"]
    # task_ids = ["TASK13"] # 最新的task,这个不应该改动...
    # task_num_iters = {"TASK0": train_dataset.num_dataset / args.train_batch_size}

    if args.visual_target == 0: # soft label
        config.v_target_size = 1601 # soft label的feature是1601维
        config.visual_target = args.visual_target # 0
    else: # 先暂时选择2048...
        config.v_target_size = 2048 # 不然的话，其中一种是feature regression，这时候feature是一个region，是2048维
        config.visual_target = args.visual_target # 1

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    if not os.path.exists(args.output_dir): # 生成输出文件夹 上面已经出现了
        os.makedirs(args.output_dir)

    task_ave_iter = {}
    task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items():
        task_ave_iter[task_id] = int(
            task_cfg[task]["num_epoch"] # 20 2
            * num_iter # 多少个batch
            * args.train_iter_multiplier # 1.0
            / args.num_train_epochs # 10.0 已改成20.0 2
        )
        task_stop_controller[task_id] = utils.MultiTaskStopOnPlateau(
            mode="max", # 这边mode必须是max
            patience=1,
            continue_threshold=0.005,
            cooldown=1,
            threshold=0.001,
        )

    task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = task_ave_iter_list[-1]
    num_train_optimization_steps = ( # NOTE 这个参数会影响下面和后面的学习率
        median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    )
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()]) # 这个预训练没有用到

    if args.dynamic_attention: # false
        config.dynamic_attention = True

    if "roberta" in args.bert_model: # # bert-base-uncased
        config.model = "roberta" # 默认之下，没有使用roberta

    if args.freeze > config.t_biattention_id[0]: # false
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention: # false
        config.with_coattention = False

    if not args.baseline: # model应该是不同的 bert-base-uncased ：） 默认是false
        model = BertForMultiModalPreTraining.from_pretrained(
            args.from_pretrained, config=config, default_gpu=default_gpu
        ) # 这里类名直接调用from_pretrained，是因为@classmethod装饰器
    else:
        model = BertForMultiModalPreTraining(config)

    model.to(device) # GPU化

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1: # 是否冻住一些权重
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if "embeddings" in name:
                bert_weight_name_filtered.append(name)
            elif "encoder" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    # if not args.from_pretrained: # 不通过这里
    if args.baseline: # 不通过这里
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        # optimizer_grouped_parameters = []
        # for key, value in dict(model.named_parameters()).items():
        #     if value.requires_grad:
        #         if key[12:] in bert_weight_name: # bert_weight_name就是前面的json文件
        #             lr = args.learning_rate * 0.1
        #         else:
        #             lr = args.learning_rate

        #         if any(nd in key for nd in no_decay):
        #             optimizer_grouped_parameters += [
        #                 {"params": [value], "lr": lr, "weight_decay": 0.0}
        #             ]

        #         if not any(nd in key for nd in no_decay):
        #             optimizer_grouped_parameters += [
        #                 {"params": [value], "lr": lr, "weight_decay": 0.01}
        #             ]
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if "vil_" in key: # multitask的权重后面有vil开头的内容
                    lr = 1e-4
                else:
                    if args.vision_scratch:
                        if key[12:] in bert_weight_name:
                            lr = base_lr
                        else:
                            lr = 1e-4
                    else:
                        lr = base_lr
                if any(nd in key for nd in no_decay): # 设定是否decay权重
                # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

        if default_gpu:
            print(
                len(list(model.named_parameters())), len(optimizer_grouped_parameters)
            )

    # set different parameters for vision branch and lanugage branch.
    # if args.fp16: # false
    #     try:
    #         # from apex.contrib.optimizers import FP16_Optimizer, FusedAdam
    #         from apex.optimizers import FusedAdam
    #         from apex.fp16_utils import FP16_Optimizer
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
    #         )

    #     optimizer = FusedAdam(
    #         optimizer_grouped_parameters,
    #         lr=base_lr, # 感觉应该是base_lr 原本是args.learning_rate
    #         bias_correction=False,
    #         # max_grad_norm=1.0,
    #     )
    #     if args.loss_scale == 0: # default是等于0
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    # else:
    optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, correct_bias=False)
        # optimizer = AdamW( # 默认使用AdamW
        #     optimizer_grouped_parameters,
        #     lr=base_lr,
        #     eps=args.adam_epsilon, # 1e-8，在VQA fine tune的过程中，保留的是1e-6，就是原本的数值
        #     betas=(0.9, 0.98), # 这个也有变化
        # )

    warmup_steps = args.warmup_proportion * num_train_optimization_steps
    warmup_scheduler = WarmupLinearSchedule( # 使用的是和trm一样的学习率策略，先增加，后减少
        optimizer, # 如果使用FP16，需要变成optimizer.optimizer！！！！
        warmup_steps=args.warmup_proportion * num_train_optimization_steps, # warmup_proportion=0.1，10%用来warmup
        t_total=num_train_optimization_steps, # 一共要优化的步骤数量
    )

    lr_reduce_list = np.array([5, 7]) # ! TODO:
    if args.lr_scheduler == "automatic":
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, cooldown=1, threshold=0.001
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=median_num_iter * args.num_train_epochs
        )
    elif args.lr_scheduler == "cosine_warm":
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=median_num_iter * args.num_train_epochs
        )
    elif args.lr_scheduler == "mannul":

        def lr_lambda_fun(epoch):
            return pow(0.2, np.sum(lr_reduce_list <= epoch))

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    startIterID = 0
    global_step = 0

    # model.cuda()
    # model.to(device)

    for state in optimizer.state.values(): # 这一段不是很理解
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    # if args.fp16: # false
    #     model.half()
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, enabled=args.fp16, opt_level='O1')

    # ******************************************************************** #
    # 继续train，checkpoint读取...
    # Note that we recommend restoring the model using the same opt_level.
    # Also note that we recommend calling the load_state_dict methods after amp.initialize.

    if args.resume_file != "" and os.path.exists(args.resume_file): # 如果有文件，就继续train
        checkpoint = torch.load(args.resume_file, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) + 1
        task_stop_controller = checkpoint["task_stop_controller"]
        tbLogger = checkpoint["tb_logger"]
        amp.load_state_dict(checkpoint['amp']) # apex的内容
        del checkpoint

    # ******************************************************************** #

    # 多卡训练
    if args.local_rank != -1: # 分布式训练
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model) # Pending * 这里使用的是Apex的DDP，不是原本pytorch的DDP
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # task_datasets_train，这里下面数据的读取应该是dataloader而不是dataset，之前CC是dataset有特殊原因，是因为td代替了dataloader
    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" % num_train_optimization_steps)
        # logger.info(
        # "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        #     device, n_gpu, bool(args.local_rank != -1), args.fp16
        # ) # 默认不进行分布式训练
        logger.info("***** Running training *****")
        logger.info("  Num Iters = {}".format(task_num_iters)) # train_dataset
        logger.info("  Batch size = {}".format(task_batch_size))     # // args.gradient_accumulation_steps)
        logger.info("  Num steps = {}".format(num_train_optimization_steps))

    task_iter_train = {name: None for name in task_ids}
    task_count = {name: 0 for name in task_ids}
    for epochId in tqdm(range(int(args.start_epoch), int(args.num_train_epochs)), desc="Epoch"): # 某一个epoch，start_epoch的default是0
        model.train()
        # torch.autograd.set_detect_anomaly(True) # 这个指令如果发现问题，会直接中断程序
        # for step, batch in enumerate(tqdm(task_dataloader_train["TASK1"], desc="Train Step")): # 这个batch就是dataloader里面__iter__的内容 # train_dataset
        # for step in tqdm(range(task_num_iters["TASK1"]), desc="Train Step"):
        for step in tqdm(range(median_num_iter), desc="Train Step"):
            # 正常应该是dataloader里面读batch，但是这边是读取datasets的内容，和train taks的代码不同。是因为这个直接处理了
            # 比较特殊？？...
            # iterId = startIterID + step + (epochId * len(task_dataloader_train["TASK1"])) # # train_dataset，得到某一个迭代的次数
            iterId = startIterID + step + (epochId * median_num_iter)
            first_task = True
            for task_id in task_ids:
                is_forward = False
                if (not task_stop_controller[task_id].in_stop) or ( # 判断是否在stop mode
                    iterId % args.train_iter_gap == 0 # 4 目的在于，在stop下面，经过一些iter之后，也可以正常运行
                ):
                    is_forward = True

                if is_forward:
                    # batch内容：features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id
                    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
                        task_iter_train[task_id] = iter(task_dataloader_train[task_id])
                    task_count[task_id] += 1
                    # GET THE BATCH
                    batch = task_iter_train[task_id].next()
                    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch) # cuda一下
                    features, spatials, image_mask, question, lm_label_ids, input_mask, segment_ids, co_attention_mask, is_next, image_label, image_target = (
                        batch
                    )

                    # args.objective的默认是0，但感觉1比较靠谱
                    # is_next == 0是match，1是不align
                    # 0: with ICA loss, \
                    # 1: with ICA loss, for the not aligned pair, no masking objective, \
                    if args.objective == 1:
                        image_label = image_label * (is_next == 0).long().unsqueeze(1)
                        image_label[image_label == 0] = -1 # 没有alignment的，后面loss也不会计算

                        lm_label_ids = lm_label_ids * (is_next == 0).long().unsqueeze(1)
                        lm_label_ids[lm_label_ids == 0] = -1 # 没有alignment的，后面loss也不会计算

                    # for task token
                    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:])) # int(task_id[4:] = 11 如果原本TASK11

                    if args.task_specific_tokens:
                        lm_task_token = lm_label_ids.new().resize_(lm_label_ids.size(0), 1).fill_(-1)
                        lm_label_ids = torch.cat([lm_label_ids[:, 0:1], lm_task_token, lm_label_ids[:, 1:]], dim=1)

                    # model的输入还有待商榷
                    masked_loss_t, masked_loss_v, next_sentence_loss = model( # # torch.Size([1])
                        question,
                        features,
                        spatials, # 这里spatial是image location
                        segment_ids, # token_type_ids
                        input_mask, # 有用信息对应mask = 1，padding的mask = 0
                        image_mask, # 有用信息对应mask = 1，padding的mask = 0
                        lm_label_ids, # pending，这个是mlm的正确答案，非答案部分是-1，-1会被之后的loss function忽略 需要有；mlm任务信号
                        image_label, # pending image_label代表当前内容是不是应该在model的loss函数里面进行mrm的计算 需要有；mrm任务信号
                        image_target, # pending mrm的image正确答案，上面一行是代表loss function是否关注该内容，而text是将二者合一，所以只有一个
                        is_next, # pending 下一个句子，不match是1.match是0，这个是用于NSP任务的信号，需要有
                        task_tokens,
                    )

                    if args.objective == 2: # 没有alignment的loss
                        next_sentence_loss = next_sentence_loss * 0

                    if args.only_ans_mlm: # 没有alignment的loss
                        next_sentence_loss = next_sentence_loss * 0

                    masked_loss_v = masked_loss_v * args.img_weight
                    loss = masked_loss_t + masked_loss_v + next_sentence_loss # # torch.Size([1])

                    if n_gpu > 1:
                        loss = loss.mean()
                        masked_loss_t = masked_loss_t.mean()
                        masked_loss_v = masked_loss_v.mean()
                        next_sentence_loss = next_sentence_loss.mean()

                    loss = loss * loss_scale[task_id] # ? FIXME: pending..... 应该是动态损失放大，没用...
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps # ! NOTE 不是很确定

                    if args.fp16: # false
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward() # 误差回传

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # if args.fp16: # false   # ??????? TODO:
                        #     # lr_this_step = args.learning_rate * warmup_linear(
                        #     #     global_step / num_train_optimization_steps,
                        #     #     args.warmup_proportion,
                        #     # )
                        #     lr_this_step = get_lr_sched(global_step, args, num_train_optimization_steps, warmup_steps)
                        #     for param_group in optimizer.param_groups:
                        #         param_group["lr"] = lr_this_step

                        # scheduler.step() # 按照scheduler，更新现有的学习率
                        optimizer.step()
                        # optimizer.zero_grad() # 每次换新的batch，都需要清零梯度
                        # 当多个模型使用同一个优化器时，二者是不同的，此时需要根据实际情况选择梯度的清除方式。
                        model.zero_grad()
                        if first_task and (
                            global_step < warmup_steps
                            or args.lr_scheduler == "warmup_linear"
                        ):
                            warmup_scheduler.step() # 这一步做的事情和上面fp16之下做的事情一样...
                        if first_task:
                            global_step += 1
                            first_task = False
                            # global_step += 1

                        # 不用改
                        if default_gpu:
                            tbLogger.step_train_CC( # 这里记录loss放在这里是合理的，因为放在前面，在梯度累加的时候
                            # 可能会有连着几个batch的loss相似的情况，导致最终画出来的loss曲线会出现短暂平台，所以应该
                            # 放在loss会实质变化的后面
                                epochId,
                                iterId,
                                float(masked_loss_t),
                                float(masked_loss_v), # torch.Size([1]) 一个step或者iter中一个batch中单个样本的损失，是平均值
                                float(next_sentence_loss),
                                optimizer.param_groups[0]["lr"],
                                task_id,
                                "train",
                            )

            if (
                step % (25 * args.gradient_accumulation_steps) == 0
                and step != 0
                and default_gpu
            ):
                tbLogger.showLossTrainCC() # pending，应该不用改

        # Do the evaluation
            flag = 0
            for task_id in task_ids:
                if (iterId != 0 and iterId % 500 == 0) or ( # 原本是iterId % task_num_iters[task_id] == 0
                    epochId == args.num_train_epochs - 1 and step == median_num_iter - 1
                ):
                    evaluate(
                        args,
                        task_dataloader_val,
                        task_stop_controller,
                        task_cfg,
                        device,
                        task_id,
                        model,
                        # task_losses,
                        epochId,
                        iterId,
                        default_gpu,
                        tbLogger,
                        n_gpu,
                        task_num_iters[task_id],
                    )
                    flag = 1
            if default_gpu and flag == 1:
                ave_score = tbLogger.showLossValCC(iterId, task_stop_controller) # Pending

        if args.lr_scheduler == "automatic":
            lr_scheduler.step(sum(val_scores.values()))
            logger.info("best average score is %3f" % lr_scheduler.best)
        elif args.lr_scheduler == "mannul":
            lr_scheduler.step()

        if epochId in lr_reduce_list: # ??
            for task_id in task_ids:
                # ? reset the task_stop_controller once the lr drop
                task_stop_controller[task_id]._reset()

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )
            output_checkpoint = os.path.join(
                savePath, "pytorch_ckpt_" + str(epochId) + ".tar"
            )
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch_id": epochId,
                    "task_stop_controller": task_stop_controller,
                    "tb_logger": tbLogger,
                    'amp': amp.state_dict(), # 存储apex的内容，比如动态损失放大的scale数值...
                },
                output_checkpoint,
            )

    if default_gpu:
        tbLogger.txt_close()

def get_lr_sched(global_step, args, num_train_optimization_steps, warmup_steps):
    # learning rate scheduling
    lr_this_step = args.learning_rate * warmup_linear(
        global_step, warmup_steps, num_train_optimization_steps)
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step

def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))

def evaluate(
    args,
    task_dataloader_val,
    task_stop_controller,
    task_cfg,
    device,
    task_id,
    model,
    # task_losses,
    epochId,
    iterId,
    default_gpu,
    tbLogger,
    n_gpu,
    task_num_iters,
):

    # Do the evaluation
    torch.set_grad_enabled(False) # 这个命令用于开关求导
    numBatches = len(task_dataloader_val[task_id]) # 原本 validation_dataset

    model.eval()
    for i, batch in enumerate(tqdm(task_dataloader_val[task_id], desc="Val Step")): # 原本 validation_dataset
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        features, spatials, image_mask, question, lm_label_ids, input_mask, segment_ids, co_attention_mask, is_next, image_label, image_target = (
            batch
        )
        # for task token
        task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:])) # int(task_id[4:] = 11 如果原本TASK11

        if args.task_specific_tokens:
            lm_task_token = lm_label_ids.new().resize_(lm_label_ids.size(0), 1).fill_(-1)
            lm_label_ids = torch.cat([lm_label_ids[:, 0:1], lm_task_token, lm_label_ids[:, 1:]], dim=1)

        batch_size = question.size(0)
        masked_loss_t, masked_loss_v, next_sentence_loss = model(
            question,
            features,
            spatials, # 这里spatial是image location
            segment_ids,
            input_mask, # 有用信息对应mask = 1，padding的mask = 0
            image_mask, # 有用信息对应mask = 1，padding的mask = 0
            lm_label_ids, # pending，这个是mlm的正确答案，非答案部分是-1，-1会被之后的loss function忽略 需要有；mlm任务信号
            image_label, # pending image_label代表当前内容是不是应该在model的loss函数里面进行mrm的计算 需要有；mrm任务信号
            image_target, # pending mrm的image正确答案，上面一行是代表loss function是否关注该内容，而text是将二者合一，所以只有一个
            is_next, # pending 下一个句子，不match是1.match是0，这个是用于NSP任务的信号，需要有
            task_tokens,
        )

        masked_loss_v = masked_loss_v * args.img_weight # args.img_weight默认是1
        loss = masked_loss_t + masked_loss_v + next_sentence_loss

        if n_gpu > 1:
            loss = loss.mean()
            masked_loss_t = masked_loss_t.mean()
            masked_loss_v = masked_loss_v.mean()
            next_sentence_loss = next_sentence_loss.mean()

        # * Done!
        if default_gpu:
            tbLogger.step_val_CC(
                epochId,
                float(masked_loss_t),
                float(masked_loss_v),
                float(next_sentence_loss),
                task_id,
                batch_size,
                "val",
            )
            sys.stdout.write("%d / %d \r" % (i, numBatches))
            sys.stdout.flush()

    # update the multi-task scheduler. NOTE
    # ! 这里先不对多任务进行更新，我们先假设second pretrain都是单任务的，所以没必要更新...
    # if iterId % task_num_iters == 0:
    #     task_stop_controller[task_id].step(tbLogger.getValScore(task_id)) # ! 因为没有score，又因为score初始化为0，所以这里只有0

    # if default_gpu:
    #     ave_score = tbLogger.showLossValCC(iterId, task_stop_controller) # pending

    torch.set_grad_enabled(True)
    model.train()


if __name__ == "__main__":

    main()
