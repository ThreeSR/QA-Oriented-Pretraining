# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# I have imported a new class LoadSecondPretrainDatasets for second pre-train, same as LoadDatasets
# in task_utils2.py.

from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}

def LoadSecondPretrainDatasets(args, task_cfg, ids, split="trainval"): # 这里以VQA为例,task_cfg就是yaml文件

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    ) # 分词器

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id # TASK1
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # H5 file is expected to have a column named "image_id", and another column
    # named "features".
    # initilzie the feature reader
    # "--in_memory" is default=False
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            ) # 这是一个读取lmdb文件的函数
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {} # 字典
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id # TASK1
        task_name = task_cfg[task]["name"] # VQA
        task_name = task_name + "_Second"
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps # 应该是确认每一步的batchsize，
        # gradient_accumulation_steps的default是1，所以batch_size暂且是128
        num_workers = args.num_workers # default = 16
        if args.local_rank != -1: # 分布式
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        # num_workers = int(num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        ) # loading VQA dataset with batch size 128

        task_datasets_train[task] = None # task = TASK1 这一步是新建一个字典
        if "train" in split: # split="trainval"
            # DatasetMapTrain[VQA] = "VQA": VQAClassificationDataset
            task_datasets_train[task] = DatasetMapTrain[task_name](
                args,
                task=task_cfg[task]["name"], # VQA
                dataroot=task_cfg[task]["dataroot"], # dataroot: /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"], # 这个没有
                split=task_cfg[task]["train_split"], # trainval
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ], # /dvmm-filer3a/users/rui/multi-task/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ], # 没有
                tokenizer=tokenizer, # BertTokenizer.from_pretrained，分词器
                bert_model=args.bert_model, #  default="bert-base-uncased"
                clean_datasets=args.clean_train_sets, # default=True
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"], # 23
                max_region_num=task_cfg[task]["max_region_num"], # 101
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = DatasetMapEval[task_name]( # train这个名字不重要，重要的是split。原本也是同一个文件，这里split不同上面
                args,
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"], # 无
                split=task_cfg[task]["val_split"], # split不同！！ minval
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,  # default="bert-base-uncased"
                clean_datasets=args.clean_train_sets, # default=True
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"], # 23
                max_region_num=task_cfg[task]["max_region_num"], # 101
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task]) # VQAClassificationDataset，随机抽取
                # RandomSampler就是根据数据集里面数据的数量，随机生成抽取数据的index列表
            else: # 多卡
                # TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task]) # 数据集分布在不同的卡上
            # 从上面已经处理之后的dataset得到dataloader
            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task], # VQAClassificationDataset
                sampler=train_sampler, # RandomSampler
                batch_size=batch_size, # 128
                num_workers=num_workers, # 16 这个参数就是在数据读取过程中，预存取在RAM中的内容；如果worker多，更多
                # 内容存在RAM中，但也加剧CPU的负担；好处是更快，但也要根据实际CPU核心数决定
                pin_memory=True, # 原本是true
                # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，
                # 则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些
            )

            task_num_iters[task] = len(task_dataloader_train[task]) # 这个就是多少个batch，就是要iter多少次
            task_batch_size[task] = batch_size

        if "val" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=2, # 原本是2
                pin_memory=True, # 原本是true
            )

    return (
        task_batch_size, # 128
        task_num_iters, # ...
        task_ids, # [TASK1]
        task_datasets_train, # VQAClassificationDataset
        task_datasets_val, # almost same as train but split is different
        task_dataloader_train,
        task_dataloader_val,
    )


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    # if task_id == "TASK4" or task_id == "TASK17":
    if task_id == "TASK3" or task_id == "TASK13":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = task_losses[task_id](vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    if task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = task_losses[task_id](vil_prediction_gqa, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction_gqa, target).sum()

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return float(loss), float(batch_score), batch_size


def ForwardModelsTrain(
    args,
    task_cfg,
    device,
    task_id,
    task_count,
    task_iter_train,
    task_dataloader_train,
    model,
    task_losses,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_id == "TASK3" or task_id == "TASK13":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )
        co_attention_mask = co_attention_mask.view(
            rbatch_size,
            co_attention_mask.size(2),
            co_attention_mask.size(3),
            co_attention_mask.size(4),
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:])) # int(task_id[4:] = 11 如果原本TASK11
    # task_tokens就是bs个id组成的列向量，比如bs * 1的11列向量
    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier": # QA类任务
        loss = task_losses[task_id](vil_prediction, target) # BCEWithLogitLoss = sigmoid + bce-loss
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(
            batch_size # ! pending 这边score的计算就是总的正确个数除以总个数,我个人认为这不是VQA中提到的VQA score的算法，这里只考虑了最大值
        )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = task_losses[task_id](vil_prediction_gqa, target) # target bs * 3129
        loss = loss.mean() * target.size(1) # 这个就是直接和target的第二个维度相乘，这里是 * 3129
        batch_score = compute_score_with_logits(
            vil_prediction_gqa, target
        ).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(
            vil_binary_prediction, target
        ).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(
            vil_tri_prediction, target
        ).sum() / float(batch_size)

    return loss, batch_score


def LoadLosses(args, task_cfg, task_ids):

    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="trainval"): # 这里以VQA为例,task_cfg就是yaml文件

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    ) # 分词器

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id # TASK1
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # H5 file is expected to have a column named "image_id", and another column
    # named "features".
    # initilzie the feature reader
    # "--in_memory" is default=False
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            ) # 这是一个读取lmdb文件的函数
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id # TASK1
        task_name = task_cfg[task]["name"] # VQA
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps # 应该是确认每一步的batchsize，
        # gradient_accumulation_steps的default是1，所以batch_size暂且是128
        num_workers = args.num_workers # default = 16
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        # num_workers = int(num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        ) # loading VQA dataset with batch size 128

        task_datasets_train[task] = None # task = TASK1
        if "train" in split: # split="trainval"
            # DatasetMapTrain[VQA] = "VQA": VQAClassificationDataset
            task_datasets_train[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"], # VQA
                dataroot=task_cfg[task]["dataroot"], # dataroot: /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"], # 这个没有
                split=task_cfg[task]["train_split"], # trainval
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ], # /dvmm-filer3a/users/rui/multi-task/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ], # 没有
                tokenizer=tokenizer, # BertTokenizer.from_pretrained，分词器
                bert_model=args.bert_model, #  default="bert-base-uncased"
                clean_datasets=args.clean_train_sets, # default=True
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"], # 23
                max_region_num=task_cfg[task]["max_region_num"], # 101
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task]) # VQAClassificationDataset，随机抽取
            else:
                # TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task])
            # 从上面已经处理之后的dataset得到dataloader
            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task], # VQAClassificationDataset
                sampler=train_sampler, # RandomSampler
                batch_size=batch_size, # 128
                num_workers=num_workers, # 16
                pin_memory=True,
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True,
            )

    return (
        task_batch_size, # 128
        task_num_iters, # ...
        task_ids, # TASK1
        task_datasets_train, # VQAClassificationDataset
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, ids):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels):  # vil_prediction, target
    logits = torch.max(logits, 1)[1].data  # argmax torch.max(a, 1): 返回每一行的最大值，且返回索引（返回最大元素在各行的列索引）
    # troch.max()[1]， 只返回最大值的每个索引，总的来说，返回每一行即每一个sample的最大值的索引
    one_hots = torch.zeros(*labels.size()).cuda() # bs * 3129
    one_hots.scatter_(1, logits.view(-1, 1), 1) # 第一个是dim，后两个是labels and scores，这样dim下，可以和原本target贴合
    scores = one_hots * labels # ! element-wise product
    return scores # bs * 3129


def EvaluatingModel(
    args,
    task_cfg,
    device,
    task_id,
    batch,
    model,
    task_dataloader,
    task_losses,
    results,
    others,
):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_id == "TASK3" or task_id == "TASK13":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )
    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )
        co_attention_mask = co_attention_mask.view(
            rbatch_size,
            co_attention_mask.size(2),
            co_attention_mask.size(3),
            co_attention_mask.size(4),
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    with torch.no_grad():
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

    if task_cfg[task_id]["type"] == "VL-classifier":
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": task_dataloader[task_id].dataset.label2ans[
                        logits[i].item()
                    ],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction_gqa, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": task_dataloader[task_id].dataset.label2ans[
                        logits[i].item()
                    ],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return float(loss), float(batch_score), batch_size, results, others
