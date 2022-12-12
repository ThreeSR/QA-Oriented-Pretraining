# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

# 多卡训练，使用mp会报错...没有用这个
import torch.multiprocessing as mp

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)

from vilbert.optimization import RAdam
from vilbert.task_utils import (
    LoadDatasets,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
)
from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

import vilbert.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument( # ! 暂时使用2048的新vilbert的权重，或者使用second pretrain之后的权重！！
        "--from_pretrained",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/pretrain/pretrained_model2.bin",
        default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/VQA-GenomeQA-GQA_bert_base_6layer_6conect-multi_task_G1_model2/pytorch_model_3.bin",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/bert_base_6layer_6conect-VQA_VGQA_GQA_SecondPretrain2/pytorch_model_1.bin",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/bert_base_6layer_6conect-VGQA_SecondPretrain3/pytorch_model_1.bin", # VQA_SecondPretrain4 是常用的
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/bert_base_6layer_6conect-VQA_VGQA_GQA_SecondPretrain/pytorch_model_1.bin",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/VQA-GenomeQA-GQA_bert_base_6layer_6conect-VQA_finetune_from_second_pretrain_G1_2/pytorch_model_9.bin",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/VQA-GenomeQA-GQA_bert_base_6layer_6conect-multi_task_G1_model/pytorch_model_8.bin",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/VQA-GenomeQA-GQA_bert_base_6layer_6conect-GQA_VGQA_G1_second_pretrain_G1/pytorch_model_17.bin",
        # default="/home/rui/code/vilbert-multi-task/vilbert-multi-task/save/VQA-GenomeQA-GQA_bert_base_6layer_6conect-VQA_G1_second_pretrain_G1/pytorch_model_4.bin",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument( # ! NOTE 默认epoch是20
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
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
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument( # * TODO:
        "--save_name",
        # default="VQA_finetune_from_first_pretrain_VQA_1601",
        # default="VQA_finetune_from_second_pretrain_VQA20",
        # default="GQA_finetune_from_second_pretrain_GQA5",
        # default="VQA_finetune_from_first_pretrain_cheat_id",
        # default="GQA_finetune_from_first_pretrain_NCCL_Test",
        # default="VQA_finetune_from_first_pretrain_VQA6",
        # default = "VQA_finetune_from_first_pretrain_G1_VQA14",
        # default = "GQA_finetune_from_first_pretrain_G1_GQA15",
        default = "VGQA_finetune_from_first_pretrain_G1_VGQA17",
        # default = "VQA_G1_second_pretrain_G1",
        # default = "VQA_finetune_from_second_pretrain_G1_2_VQA7",
        # default = "GQA_VGQA_G1_second_pretrain_G1",
        # default = "VGQA_finetune_from_G1_second_pretrain_VGQA4",
        # default = "GQA_finetune_from_G1_second_pretrain_GQA4",
        # default = "VQA_finetune_from_G1_second_pretrain_VQA7",
        # default = "GQA_finetune_from_first_pretrain_GQA_Test",
        # default = "GQA_finetune_from_first_pretrain_GQA3",
        # default = "VGQA_finetune_from_first_pretrain_VGQA3",
        # default = "VGQA_finetune_from_second_pretrain_VGQA5",
        # default = "multi_task_G1_model2",
        # default = "VQA_finetune_from_VQA_G1_second_pretrain_G1_VQA13",
        # default = "GQA_finetune_from_GQA_VGQA_G1_second_pretrain_G1_GQA20",
        # default = "VGQA_finetune_from_GQA_VGQA_G1_second_pretrain_G1_VGQA7",
        # default = "GQA_Test",
        type=str,
        help="save name for training."
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--optim", default="AdamW", type=str, help="what to use for the optimization."
    )
    parser.add_argument(
        "--tasks", default="", type=str, help="1-2-3... training task separate by -"
    ) # 代表进行fine tune哪一个任务
    parser.add_argument(
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument(
        "--vision_scratch",
        action="store_true",
        help="whether pre-trained the image or not.",
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler",
        default="mannul",
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument( # * TODO: 这里暂时取1，之后取完特征，再改成0
        "--visual_target",
        default=1, # !!!!!!!!!!!!!!!!!!!!!
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        help="whether to use task specific tokens for the multi-task learning.",
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

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    else:
        from vilbert.vilbert import BertConfig # 现在已经改成承接second pretrain的内容
        from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split("-")): # 1-2-3...输入任务，假设这时候只有任务一，在进行fine tune
        task = "TASK" + task_id  # TASK1
        name = task_cfg[task]["name"] # VQA
        task_names.append(name)
        task_lr.append(task_cfg[task]["lr"])

    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        loss_scale[task] = task_lr[i] / base_lr # 1

    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timeStamp = (
        "-".join(task_names)
        + "_"
        + args.config_file.split("/")[1].split(".")[0] # VQA_bert_base_6layer_6conect-VQA_finetune_from_second_pretrain_VQA
        + prefix
    )
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(
        open("config/" + args.bert_model + "_weight_name.json", "r")
    )
    # ******************************************************************** #
    # 多卡训练
    if args.local_rank == -1 or args.no_cuda:
        # print(torch.cuda.is_available()) # true,之前false，不知道为什么
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

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank() # * rank会自己得到
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)
    # 这里load data，包含了所有情况，在数据文件里面，有对不同数据文件的处理
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = LoadDatasets(
        args, task_cfg, args.tasks.split("-")
    )

    logdir = os.path.join(savePath, "logs")
    tbLogger = utils.tbLogger(
        logdir,
        savePath,
        task_names,
        task_ids,
        task_num_iters,
        args.gradient_accumulation_steps,
    )

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    if not os.path.exists(args.output_dir): # 生成输出文件夹
        os.makedirs(args.output_dir)

    task_ave_iter = {}
    task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items():
        task_ave_iter[task_id] = int(
            task_cfg[task]["num_epoch"] # 20 6
            * num_iter
            * args.train_iter_multiplier
            / args.num_train_epochs # 20 6
        )
        task_stop_controller[task_id] = utils.MultiTaskStopOnPlateau(
            mode="max",
            patience=1,
            continue_threshold=0.005,
            cooldown=1,
            threshold=0.001,
        )

    task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = task_ave_iter_list[-1]
    num_train_optimization_steps = (
        median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    )
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    if args.dynamic_attention:
        config.dynamic_attention = True
    if "roberta" in args.bert_model:
        config.model = "roberta"

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )
    else:
        model = VILBertForVLTasks.from_pretrained( # 一般情况下，是使用vilbert
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )

    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
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

    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = base_lr
                    else:
                        lr = 1e-4
                else:
                    lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, correct_bias=False)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)

    warmpu_steps = args.warmup_proportion * num_train_optimization_steps

    if args.lr_scheduler == "warmup_linear":
        warmup_scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmpu_steps, t_total=num_train_optimization_steps
        )
    else:
        warmup_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmpu_steps)

    lr_reduce_list = np.array([5, 7])
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
    start_epoch = 0

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, enabled=args.fp16, opt_level='O1')

    # ******************************************************************** #
    # 继续train，checkpoint读取...
    # Note that we recommend restoring the model using the same opt_level.
    # Also note that we recommend calling the load_state_dict methods after amp.initialize.

    if args.resume_file != "" and os.path.exists(args.resume_file): # 继续train机制
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
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) + 1
        task_stop_controller = checkpoint["task_stop_controller"]
        tbLogger = checkpoint["tb_logger"]
        amp.load_state_dict(checkpoint['amp']) # apex的内容
        del checkpoint
    # ******************************************************************** #
    # 多卡训练
    if args.local_rank != -1: # 如果是使用Apex
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True) # * 这里使用的是Apex的DDP，不是原本pytorch的DDP

    elif n_gpu > 1: # ! 如果是使用一般的多卡，现在应该都是用DDP了，但这个DDP和上面apex的DDP不同，这个要注意！！
        model = torch.nn.DataParallel(model)
    # ******************************************************************** #
    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" % num_train_optimization_steps)

    task_iter_train = {name: None for name in task_ids}
    task_count = {name: 0 for name in task_ids}
    for epochId in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"): # tqdm模块——进度条配置，就是可以展现进度条
        model.train()
        # if not args.fp16: model.float()
        # torch.autograd.set_detect_anomaly(True)
        for step in tqdm(range(median_num_iter), desc="Train Step"):
            iterId = startIterID + step + (epochId * median_num_iter)
            first_task = True
            for task_id in task_ids:
                is_forward = False
                if (not task_stop_controller[task_id].in_stop) or (
                    iterId % args.train_iter_gap == 0
                ):
                    is_forward = True

                if is_forward:
                    loss, score = ForwardModelsTrain(
                        args,
                        task_cfg,
                        device,
                        task_id,
                        task_count,
                        task_iter_train,
                        task_dataloader_train,
                        model,
                        task_losses,
                    )

                    loss = loss * loss_scale[task_id]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16: # false
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward() # 误差回传

                    # loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # if args.fp16:
                        #     lr_this_step = args.learning_rate * warmup_linear(
                        #         global_step / num_train_optimization_steps,
                        #         args.warmup_proportion,
                        #     )
                        #     for param_group in optimizer.param_groups:
                        #         param_group["lr"] = lr_this_step

                        optimizer.step()
                        model.zero_grad()
                        if first_task and (
                            global_step < warmpu_steps
                            or args.lr_scheduler == "warmup_linear"
                        ):
                            warmup_scheduler.step()
                        if first_task:
                            global_step += 1
                            first_task = False

                        if default_gpu:
                            tbLogger.step_train(
                                epochId,
                                iterId,
                                float(loss),
                                float(score),
                                optimizer.param_groups[0]["lr"],
                                task_id,
                                "train",
                            )

            if "cosine" in args.lr_scheduler and global_step > warmpu_steps:
                lr_scheduler.step()

            if (
                step % (20 * args.gradient_accumulation_steps) == 0
                and step != 0
                and default_gpu
            ):
                tbLogger.showLossTrain()

            # decided whether to evaluate on each tasks.
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
                        task_losses,
                        epochId,
                        default_gpu,
                        tbLogger,
                        task_num_iters[task_id],
                        iterId,
                    )

        if args.lr_scheduler == "automatic":
            lr_scheduler.step(sum(val_scores.values()))
            logger.info("best average score is %3f" % lr_scheduler.best)
        elif args.lr_scheduler == "mannul":
            lr_scheduler.step()

        if epochId in lr_reduce_list:
            for task_id in task_ids:
                # reset the task_stop_controller once the lr drop
                task_stop_controller[task_id]._reset()

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            ) # 这个bin文件只有模型权重，保留多个；下面的checkpoint文件比这个权重文件更大，只留一个...
            output_checkpoint = os.path.join(savePath, "pytorch_ckpt_latest.tar") # checkpoint文件只保留一个
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch_id": epochId,
                    "task_stop_controller": task_stop_controller,
                    "tb_logger": tbLogger,
                    'amp': amp.state_dict(), # 存储apex的内容，比如动态损失放大的scale数值...
                },
                output_checkpoint,
            )
    tbLogger.txt_close()


def evaluate(
    args,
    task_dataloader_val,
    task_stop_controller,
    task_cfg,
    device,
    task_id,
    model,
    task_losses,
    epochId,
    default_gpu,
    tbLogger,
    task_num_iters,
    iterId,
):

    model.eval()
    for i, batch in enumerate(tqdm(task_dataloader_val[task_id], desc="Val Step")):
        loss, score, batch_size = ForwardModelsVal(
            args, task_cfg, device, task_id, batch, model, task_losses
        )
        tbLogger.step_val(
            epochId, float(loss), float(score), task_id, batch_size, "val"
        )
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()

    # update the multi-task scheduler. NOTE
    if iterId % task_num_iters == 0:
        task_stop_controller[task_id].step(tbLogger.getValScore(task_id))

    score = tbLogger.showLossVal(task_id, task_stop_controller)
    model.train()


if __name__ == "__main__":

    main()
