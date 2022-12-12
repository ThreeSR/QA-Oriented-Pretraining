# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging
import pickle
import random
import tqdm
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import Dataset

from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected): # 自己写的一个判断是否相等的函数
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


# def _create_entry(question, answer):
#     # 原本answers：{'image_id': 458752, 'labels': [1164], 'scores': [1], 'question_id': 458752000}
#     answer.pop("image_id")
#     answer.pop("question_id")
#     # 现在：{'labels': [1164], 'scores': [1]}
#     # 下面entry需要不同id，应该是因为一张图可以对应多个问题
#     # question: {'image_id': 262144, 'question': 'Is the ball flying towards the batter?', 'question_id': 262144000}
#     entry = {
#         "question_id": question["question_id"], # 262144000
#         "image_id": question["image_id"], # 262144
#         "question": question["question"], # 'Is the ball flying towards the batter?'
#         "answer": answer, # 前面两个pop的原因是，question上面的内容和pop的内容重叠了，所以pop掉重复的部分
#         # answer : {'labels': [1164], 'scores': [1]}
#     } # 字典
#     return entry

def _create_entry(args, question, answer, dataroot, label2ans): # ！！直接使用读取好的label2ans省时间！！
    # 原本answers：{'image_id': 458752, 'labels': [1164], 'scores': [1], 'question_id': 458752000}
    answer.pop("image_id")
    answer.pop("question_id")
    # 现在：{'labels': [1164], 'scores': [1]}
    # 下面entry需要不同id，应该是因为一张图可以对应多个问题
    # question: {'image_id': 262144, 'question': 'Is the ball flying towards the batter?', 'question_id': 262144000}
    entry = {
        "question_id": question["question_id"], # 262144000
        "image_id": question["image_id"], # 262144
        "question": question["question"], # 'Is the ball flying towards the batter?'
        "answer": answer, # 前面两个pop的原因是，question上面的内容和pop的内容重叠了，所以pop掉重复的部分
        # answer : {'labels': [1164], 'scores': [1]}
    } # 字典
    # 此时还需要得到答案的原本文本文件trainval_label2ans.pkl
    # label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl") # len是3129
    # # label2path = cPickle.load(open(label2ans_path), "rb")
    # with open(label2ans_path, "rb") as f:
    #     label2ans = pickle.load(f) # 就是一个字典，key是int，value是'net' ... {...,1164:'net',...}
    scores = sorted(answer['scores'], reverse=True)
    # print(scores)
    element = scores[0]
    idx = answer['scores'].index(element) # 得到1对应的下标,或者是得到最大值所对应的下标
    label = answer['labels'][idx] # 此时是一个数值，是1164
    answer_text = label2ans[label] # 得到真正的答案对应的原文本，这个原文本可能长度是1或者2或者3，'net' or "pink and yellow"
    if not args.sep: # * 不使用sep的时候，文本直接加在一起
        entry["question"] = entry["question"] + ' ' + answer_text # 两个string相加，直接把答案插在问题后面
        # entry["question"] = entry["question"] + ' ' + str(label) # 两个string相加，直接把答案插在问题后面
        # entry["question"] = answer_text # 这个时候只有答案
    else:
        # entry["answer_text"] = answer_text
        entry["answer_text"] = str(label)
    # 'Is the ball flying towards the batter?' + 空格 +"pink and yellow"  --->>>空格问题pending...
    # 'Is the ball flying towards the batter? pink and yellow'
    return entry

def iou(anchors, gt_boxes): # pending...估计要改，归一化应该不影响iou的计算，所以输入不用变，就是函数里面变一变就好
    """
    anchors和gt_boxes是一样的，应该都只有四维，都是坐标x y x y形式
    anchors: (N, 4) ndarray of float，现在大小应该是（N + 1） * 5
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    # # I think since the first row is special, we need to remove it.
    anchors = anchors[1:, :]
    gt_boxes = gt_boxes[1:, :]

    N = anchors.shape[0]
    K = gt_boxes.shape[0] # N == K ?? yes

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1) # 为什么加1？？？
    ).reshape(1, K) # reshape只是让两个相同的内容更好地被处理

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 5), K, axis=1) # repeat函数功能：对数组中的元素进行连续重复复制
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 5), N, axis=0) # 这里应该是5不是4

    iw = (
        np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
        - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    ) # 为什么加1？？？？？# N*N
    iw[iw < 0] = 0 # 不相交等于0

    ih = (
        np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
        - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    ) # N*N
    ih[ih < 0] = 0
    # anchors_area N*1, gt_boxes_area 1*N，下面两个相加，直接广播，变成N*N
    ua = anchors_area + gt_boxes_area - (iw * ih) # A + B - C   这里的*应该就是element-wise相乘的意思
    overlaps = iw * ih / ua # C / (A + B - C)，IOU就是两个框的公共区域面积除以两个框一共占有的面积，这个面积要扣去公共面积

    return overlaps # N*N，每一行代表这个bbox和其他bbox之间的overlap大小

def _load_dataset(args, dataroot, name, clean_datasets, label2ans): # 基本上都是读取问题，读取答案，进行排序，进行match
    # dataroot /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
    # name trainval or minval; clean_datasets = _cleaned
    """
    Load entries
    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minval'
    """
    # 下面各种if下，不同的情况，反应的是不同的数据训练检测的分割方式
    if name == "train" or name == "val": # not pass :(
        question_path = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name
        ) # 这里的问题是文本文件，还没有tokenize
        questions = sorted(
            json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
        ) # 按照question id对question进行排序
        answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name) # 获取到问题对应的答案
        answers = cPickle.load(open(answer_path, "rb")) # 因为是pkl文件，所以是pickle进行load
        answers = sorted(answers, key=lambda x: x["question_id"]) # 这样排序之后，答案和问题相互match

    elif name == "trainval": # here we go :) 训练
        question_path_train = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        ) # question_path_train = /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
        # v2_OpenEnded_mscoco_train2014_questions.json
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        ) # 按照question id，对问题进行排序
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        # /dvmm-filer3a/users/rui/multi-task/datasets/VQA/cache/train_target.pkl
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"]) # 这样一来，问题和答案对应
        # 下面是val部分，same as train
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        # 得到所有的问题和答案
        questions = questions_train + questions_val[:-3000] # 不知为何这么操作？？
        answers = answers_train + answers_val[:-3000]

    elif name == "minval":  # here we go too :) 测试;这一部分和上面trainval是呼应的
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_val[-3000:] # 这里呼应上面trainval，人为隔开训练和测试集
        answers = answers_val[-3000:] # 事实证明，这个测试集确实是3000个数据

    elif name == "test":
        question_path_test = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % "test"
        )
        questions_test = sorted(
            json.load(open(question_path_test))["questions"],
            key=lambda x: x["question_id"],
        )
        questions = questions_test

    elif name == "mteval":
        question_path_train = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        )
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])

        questions = questions_train
        answers = answers_train
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    elif name == "mteval":
        entries = []
        remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
        remove_ids = [int(x) for x in remove_ids] # 数据类型转换

        for question, answer in zip(questions, answers):
            if int(question["image_id"]) in remove_ids:
                entries.append(_create_entry(question, answer))
    else: # here we go :)
        assert_eq(len(questions), len(answers)) # 看是否相等
        entries = []
        remove_ids = []
        if clean_datasets: # here we go :)
            remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy")) # 这个npy是image_id
            remove_ids = [int(x) for x in remove_ids] # 原本是string，现在int化
        # test = 300
        for question, answer in tqdm(zip(questions, answers), desc="Create Entry", total=len(questions)):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue # 这部分内容删去，不使用。这就是clean掉的部分，所以cleaned用意在此
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            # test = test - 1
            # if test == 0: break
            if len(answer['scores']) == 0: continue # 这一行代码的目的是滤除没有答案的问题，这部分占比很小，大约2%
            entries.append(_create_entry(args, question, answer, dataroot, label2ans)) # 这就是留下的，最后使用的数据部分

    return entries # 字典，里面存着问答内容，图像id，问题id。简单来说就是question id + V + Q + A
            # 注意！！：这里的entries是还没有token的内容，就是原始文本

# 在送进dataloader之前，需要自己定义自己想要送进去的数据集，这里的父类是pytorch自己的Dataset类型
class VQALoader(Dataset): # pending，看是否改名
    def __init__(
        self,
        args,
        task, # VQA
        dataroot, # /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
        annotations_jsonpath,
        split, # trainval或者minval
        image_features_reader, # /dvmm-filer3a/users/rui/multi-task/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb
        gt_image_features_reader,
        tokenizer, # BertTokenizer.from_pretrained，分词器
        bert_model, # default="bert-base-uncased"
        clean_datasets, # default=True
        padding_index=0,
        max_seq_length=16, # 这个应该还是23
        max_region_num=100,
    ):
        super().__init__()
        self.split = split # trainval or minval
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl") # 潜在的3000多个答案
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label) # 3129
        self._max_region_num = max_region_num # 101
        self._max_seq_length = max_seq_length # 23
        self._image_features_reader = image_features_reader # h5读取的features，lmdb文件
        self._tokenizer = tokenizer # 分词器  BertTokenizer.from_pretrained
        self._padding_index = padding_index # 0
        # !! 小心是36还是100
        imgid2img_trainval_path = os.path.join(dataroot, "cache", "Imgid2img_trainval2014_100.pkl") # !!
        self.imgid2img_trainval = cPickle.load(open(imgid2img_trainval_path, "rb"))

        self.config = args # Pending

        self.visualization = False
        self.objective = 1 # 这个需要根据pretrain_second文件里面的配置进行更改...

        clean_train = "_cleaned" if clean_datasets else "" # clean_datasets=True，选上了

        if "roberta" in bert_model: # 没有，这个不通过，没有roberta
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + clean_train
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + clean_train + '_qa' + ".pkl",
                # task + "_" + split + "_" + str(max_seq_length) + clean_train + '_answer' + ".pkl", # !!!!!!!!
            ) # 输入读取文件或者存储文件的路径
            # cache_path = /dvmm-filer3a/users/rui/multi-task/datasets/VQA/cache/VQA_trainval_23_cleaned.pkl
        if args.sep: # if use sep token
            cache_path = os.path.join(
                dataroot,
                "cache",
                # task + "_" + split + "_" + str(max_seq_length) + clean_train + '_qa' + "_sep" + "_id" + ".pkl",
                task + "_" + split + "_" + str(max_seq_length) + clean_train + '_qa' + "_sep" + ".pkl",
            )
        # 标记一个新的path，代表这个里面的question id之后有answer的原文本
        # /dvmm-filer3a/users/rui/multi-task/datasets/VQA/cache/VQA_trainval_23_cleaned_qa.pkl
        # /dvmm-filer3a/users/rui/multi-task/datasets/VQA/cache/VQA_trainval_30_cleaned_qa_sep.pkl
            if args.only_ans_mlm: # ! 目前only ans mlm是仅仅用在sep下面的...所以二者要连用！
                cache_path = os.path.join(
                    dataroot,
                    "cache",
                    task + "_" + split + "_" + str(max_seq_length) + clean_train + '_qa' + "_sep" + "_ans_mlm" + ".pkl",
                )

        if not os.path.exists(cache_path): # 这边的操作就是，把上面的原始文本token化
            self.entries = _load_dataset(args, dataroot, split, clean_datasets, self.label2ans) # 这就是前面的函数，得到ID + VQA
            self.num_caps = len(self.entries) # 得到一共多少question，多少文本
            if args.sep:
                self.tokenize_sep(max_seq_length)
            else:
                self.tokenize(max_seq_length) # 23，调用了下面的函数
            # self.tensorize() # tensor化，所以这个数据集之后需要在torch环境下才可以打开

            # for entry in tqdm(self.entries, desc="Collect Image Features"):
            #     is_next = entry['is_next']
            #     image_id = entry["image_id"] # 得到两个id
            #     # # 根据image id得到feature，触发getitem方法，读取图像特征
            #     image_feat, num_boxes, image_loc, _ = self._image_features_reader[image_id] # 101 bboxes
            #     image_feat_ori = image_feat # 留着之后求特征回归,原本的feature，没有mask
            #     # eg: array..., 101, boxes。第四个不用管，是original大小的图像特征，这个用不到，因为归一化加速模型收敛
            #     # 第一个array是各个box的features，所以是num_bbox * 2048维，比如101个bbox，每个bbox的特征是2048维向量
            #     # 第三个boxes是坐标位置信息，这个是归一化之后的信息，我们没有使用归一化之前的信息，（num-bbox + 1） * 5 （已归一化）
            #     # _max_region_num = 101
            #     overlaps = iou(image_loc, image_loc) # 经过计算，归一化与否应该不影响IOU的计算
            #     image_feat, image_loc, image_label, masked_label = self.random_region( # mrm之后的内容
            #         image_feat, image_loc, num_boxes, is_next, overlaps
            #     ) # image_label代表当前内容是不是应该在model的loss函数里面进行mrm的计算
            #     # image_label 是1 * num_bbox大小的list
            #     # tokens = self._tokenizer.add_special_tokens_single_sentence(tokens) # 加上CLS 和 SEP
            #     # input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)，这个是换过单词之后的句子

            #     # _max_region_num = 101
            #     mix_num_boxes = min(int(num_boxes), self._max_region_num) # 不能超过最多bb数量
            #     mix_boxes_pad = np.zeros((self._max_region_num, 5)) # 101 * 5，5来自于box是5维向量
            #     mix_features_pad = np.zeros((self._max_region_num, 2048)) # 2048是因为抽取的特征是2048维向量
            #     mix_features_pad_ori = np.zeros((self._max_region_num, 2048))

            #     image_mask = [1] * (int(mix_num_boxes)) # 非padding区域是1，padding区域是0，用以区分
            #     while len(image_mask) < self._max_region_num:
            #         image_mask.append(0) # 这个mask是按照bb个数算的，一个数顶一个region，代表这个region应不应该mask

            #     # added by myslef
            #     image_label = image_label[:mix_num_boxes]
            #     while len(image_label) < self._max_region_num:
            #         image_label.append(-1)

            #     entry['image_label'] = image_label # image_label的tensor在下面tensorlize里面
            #     # shuffle the image location here.
            #     # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
            #     # img_idx.append(0)
            #     # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
            #     # mix_features_pad[:mix_num_boxes] = features[img_idx]

            #     # padding，直至填满最大的region数量为止，就是在padding，在padding 0
            #     mix_boxes_pad[:mix_num_boxes] = image_loc[:mix_num_boxes] # 把有用内容装进来，多余的就是padding的
            #     mix_features_pad[:mix_num_boxes] = image_feat[:mix_num_boxes] # # 把有用内容装进来，多余的就是padding的
            #     mix_features_pad_ori[:mix_num_boxes] = image_feat_ori[:mix_num_boxes] # 留住之前的内容

            #     # tensor化
            #     features = torch.tensor(mix_features_pad).float()
            #     features_ori = torch.tensor(mix_features_pad_ori).float()
            #     image_mask = torch.tensor(image_mask).long() # 有用信息对应mask = 1，padding的mask = 0
            #     spatials = torch.tensor(mix_boxes_pad).float() # location信息，也是空间信息，spatial

            #     entry["image_target"] = features_ori
            #     entry["features"] = features
            #     entry["spatials"] = spatials
            #     entry["image_mask"] = image_mask

                # 这部分没什么用
                # co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length)) # 文本和图像互信息，101 * 23
                # target = torch.zeros(self.num_labels) # 3129个0

                # if "test" not in self.split: # ：） go go go
                #     answer = entry["answer"] # {'labels': [1164], 'scores': [1]}
                #     labels = answer["labels"] # [1164]，已经tensor化！
                #     scores = answer["scores"] # [1]
                #     if labels is not None:
                #         target.scatter_(0, labels, scores) # 这个函数在这里的意思是，将label对应的score送到target上
                    # eg：target = torch.zeros(3129)
                        #   labels = torch.tensor([1, 2])
                        #   scores = torch.tensor([1, 0.6])
                        #   target.scatter_(0, labels, scores)
                        #   print(target)
                        #   tensor([0.0000, 1.0000, 0.6000,  ..., 0.0000, 0.0000, 0.0000])
            self.tensorize() # tensor化，所以这个数据集之后需要在torch环境下才可以打开
            cPickle.dump(self.entries, open(cache_path, "wb")) # 存储已经处理好的数据为pkl文件
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb")) # 如果已经有处理好的文件，那么直接读取
            self.num_caps = len(self.entries) # 得到一共多少question，多少文本

    def tokenize(self, max_length=16): # max_length是怎么确定的？目前max length是23
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset. 在原本数据集上增加token之后的question内容
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in tqdm(self.entries, desc="Tokenizes"):
            # random_cap是为了多模态对齐任务，就是 multimodal alignment，也就是ITM
            caption, label = self.random_cap(entry["question"]) # 看是否要随机选择另外一个句子，这时候还是文本
            is_next = label
            tokens = self._tokenizer.encode(caption) # question（文本）编码，这时候从原文本变成token，这个是数字
            # 这里是为了MLM
            tokens, tokens_label = self.random_word(tokens, self._tokenizer, is_next)
            # 到位，encode实现了先toknize再变成数字的步骤
            tokens = tokens[: max_length - 2] # 空出位置
            tokens_label = tokens_label[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens) # 加入cls之类的特殊token，前面cls，后面sep

            segment_ids = [0] * len(tokens) # 这个应该就是BERT原文提到的segment embedding
            input_mask = [1] * len(tokens)

            lm_label_ids = [-1] + tokens_label + [-1] # 这一部分留存了MLM的正确答案，无关部分都是-1

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                gap = max_length - len(tokens)
                padding = [self._padding_index] * (max_length - len(tokens)) # 这个_padding_index就是pad谁，这里是0
                tokens = tokens + padding
                input_mask += padding # input_mask应该就是表明哪里是文字，文字对应1，哪里是padding，padding对应0
                segment_ids += padding
                padding2 = [-1] * gap
                lm_label_ids += padding2

            assert_eq(len(tokens), max_length) # 确认上述操作
            assert_eq(len(lm_label_ids), max_length)
            # 在原本的entry中加入新的内容，加上token之后的成果
            entry["q_token"] = tokens # 已经是最后一步，是数字了
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids
            entry['is_next'] = is_next # 这个仅仅只是一个数值，用于标记
            entry['lm_label_ids'] = lm_label_ids # 这个之后需要输入模型，这个是一个list，应该进行tensor化

    def tokenize_sep(self, max_length=16): # max_length是怎么确定的？目前max length是23
        # ! 此处的question只有question，没有answer在后面
        """
        Tokenizes the questions and answers, but they are separate.
        This will add q_token in each entry of the dataset. 在原本数据集上增加token之后的question内容
        -1 represent nil, and should be treated as padding_index in embedding
        此处的question只有question，没有answer在后面
        """
        for entry in tqdm(self.entries, desc="Tokenizes_sep"):
            # random_cap是为了多模态对齐任务，就是 multimodal alignment，也就是ITM
            if not self.config.only_ans_mlm:
                caption, answer, label = self.random_cap_sep(entry["question"], entry["answer_text"]) # 看是否要随机选择另外一个句子，这时候还是文本
                is_next = label
            else:
                caption, answer, is_next = entry["question"], entry["answer_text"], 0

            tokens_q = self._tokenizer.encode(caption) # question 编码，这时候从原文本变成token，这个是数字
            tokens_a = self._tokenizer.encode(answer) # encode answer
            len_q = len(tokens_q)
            len_a = len(tokens_a)

            # 这里是为了MLM
            if self.config.only_ans_mlm:
                tokens_label = [-1] * len_q
                tokens_a, output_label = self.random_word_only_ans_mlm(tokens_a, self._tokenizer, is_next)
                tokens_label = tokens_label + output_label
            else:
                tokens = tokens_q + tokens_a # 整条文本
                tokens, tokens_label = self.random_word(tokens, self._tokenizer, is_next)
                # random word之后，再把问题和答案拆分开来
                tokens_q = tokens[:len_q]
                tokens_a = tokens[len_q:]
                assert_eq(len(tokens_a), len_a) # testify whether it is correct or not
            # 到位，encode实现了先toknize再变成数字的步骤
            # tokens = tokens[: max_length - 3] # 空出位置，多空出一个sep位置
            # !!!!!!!!!!!!!!
            # tokens = tokens[: max_length - 3] # 空出位置，多空出一个sep位置
            # tokens_label = tokens_label[: max_length - 3] # 空出位置，多空出一个sep位置
            tokens = self._tokenizer.add_special_tokens_sentences_pair(tokens_q, tokens_a)
            # * 得到 tokens = cls + question + sep + answer + sep
            # tokens = self._tokenizer.add_special_tokens_single_sentence(tokens) # 加入cls之类的特殊token，前面cls，后面sep

            segment_ids_q = [0] * len(tokens_q) # 这个应该就是BERT原文提到的segment embedding
            segment_ids_a = [1] * len(tokens_a) # answer's segment id
            segment_ids = [0] + segment_ids_q + [0] + segment_ids_a + [1] # cls + question + sep + answer + sep
            input_mask = [1] * len(tokens)

            lm_label_ids = [-1] + tokens_label + [-1] # 这一部分留存了MLM的正确答案，无关部分都是-1
            # list 指定idx位置添加元素 guests.insert(idx,'Hu qi')
            lm_label_ids.insert(1 + len(tokens_q), -1) # 填补中间的sep对应的-1

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                gap = max_length - len(tokens)
                padding = [self._padding_index] * (max_length - len(tokens)) # 这个_padding_index就是pad谁，这里是0
                tokens = tokens + padding
                input_mask += padding # input_mask应该就是表明哪里是文字，文字对应1，哪里是padding，padding对应0
                segment_ids += padding
                padding2 = [-1] * gap
                lm_label_ids += padding2

            assert_eq(len(tokens), max_length) # 确认上述操作
            assert_eq(len(lm_label_ids), max_length)
            # 在原本的entry中加入新的内容，加上token之后的成果
            entry["q_token"] = tokens # 已经是最后一步，是数字了
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids
            entry['is_next'] = is_next # 这个仅仅只是一个数值，用于标记
            entry['lm_label_ids'] = lm_label_ids # 这个之后需要输入模型，这个是一个list，应该进行tensor化

    def tensorize(self): # 将变量tensor化

        for entry in tqdm(self.entries, desc="Tensorize"):
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids
            # added by myself
            lm_label_ids = torch.from_numpy(np.array(entry["lm_label_ids"]))
            entry["lm_label_ids"] = lm_label_ids

            if "test" not in self.split: # :) here we go
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None
    # 在python中__getitem__(self,key)方法被称为魔法方法，这个方法返回所给键对应的值
    # 在dataloader的底层实现里面，会有调用getitem的地方，总之这个就是使用dataloder之后的输出出口
    def __getitem__(self, index): # 在原本只有处理好的文本基础上，读取图像feature
        """之所以把图像抽取放在这里，是因为预抽取图像特征太浪费时间，所以在这边单个抽取

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        entry = self.entries[index] # 获取某个entry
        # image_id = entry["image_id"] # 得到两个id
        # question_id = entry["question_id"]
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id] # 根据image id得到feature
        # 图像特征前面应该有一个IMG token，但是不管是CC的原本代码还是这里，都没有加这个token
        # 我认为是在_image_features_reader里面，已经有赠加了这个token，就是g-location和g-feat，所以接下来，不用管img上面的特殊token
        # eg: array..., 101, boxes。第四个不用管，是一个什么original...
        # 第一个array是各个box的features，所以是num_bbox * 2048维，比如101个bbox，每个bbox的特征是2048维向量
        # 第三个boxes是坐标位置信息
        # _max_region_num = 101
        # mix_num_boxes = min(int(num_boxes), self._max_region_num) # 不能超过最多bb数量
        # mix_boxes_pad = np.zeros((self._max_region_num, 5)) # 101 * 5，5来自于box是5维向量
        # mix_features_pad = np.zeros((self._max_region_num, 2048)) # 2048是因为抽取的特征是2048维向量

        # image_mask = [1] * (int(mix_num_boxes)) # 非padding区域是1，padding区域是0，用以区分
        # while len(image_mask) < self._max_region_num:
        #     image_mask.append(0) # 这个mask是按照bb个数算的，一个数顶一个region，代表这个region应不应该mask

        # shuffle the image location here.
        # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
        # img_idx.append(0)
        # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
        # mix_features_pad[:mix_num_boxes] = features[img_idx]

        # padding，直至填满最大的region数量为止，就是在padding，在padding 0
        # mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes] # 把有用内容装进来，多余的就是padding的
        # mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes] # # 把有用内容装进来，多余的就是padding的
        # tensor化
        # features = torch.tensor(mix_features_pad).float()
        # image_mask = torch.tensor(image_mask).long() # 有用信息对应mask = 1，padding的mask = 0
        # spatials = torch.tensor(mix_boxes_pad).float() # location信息，也是空间信息，spatial
        is_next = entry['is_next']
        image_id = entry["image_id"] # 得到两个id
        # # 根据image id得到feature，触发getitem方法，读取图像特征


        # 如果使用10-100，以下面为准，如果使用36，以更下面为准
        # FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

        img = self.imgid2img_trainval[str(image_id).zfill(12)]
        # imgid2img_trainval['000000573195']
        # id都是唯一的，和train与val无关；格式如上，之后得到具体字典，字典的keys如下
        # odict_keys(['img_id', 'img_h', 'img_w', 'objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'num_boxes', 'boxes', 'features'])
        num_boxes = img['num_boxes']
        image_feat = img['features']
        boxes = img['boxes'] # !! 都是原坐标！！没有进行归一化，这里应该进行归一化

        # !!!!!!!!!!!!!!
        # 36 和 10-100 对应不同的键的名字  这里两个文件的width和height都是对应的，没问题，后面归一化也是对的...
        img_h, img_w = img['image_h'], img['image_w']

        # ! 归一化操作，还需要加入面积
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0]) / (img_h * img_w)
        ) # 加上面积维度, (x1, y1) & (x2, y2)

        image_location[:, (0, 2)] /= img_w
        image_location[:, (1, 3)] /= img_h

        image_loc = image_location

        # image_feat, num_boxes, image_loc, _ = self._image_features_reader[image_id] # 101 bboxes
        image_feat_ori = image_feat # 留着之后求特征回归,原本的feature，没有mask
        # eg: array..., 101, boxes。第四个不用管，是original大小的图像特征，这个用不到，因为归一化加速模型收敛
        # 第一个array是各个box的features，所以是num_bbox * 2048维，比如101个bbox，每个bbox的特征是2048维向量
        # 第三个boxes是坐标位置信息，这个是归一化之后的信息，我们没有使用归一化之前的信息，（num-bbox + 1） * 5 （已归一化）
        # _max_region_num = 36

        overlaps = iou(image_loc, image_loc) # 经过计算，归一化与否应该不影响IOU的计算

        # ! NOTE 这里记得要手动变化是否使用IOU，现在先不使用
        set_zero = np.zeros_like(overlaps)
        overlaps = set_zero # ! 姑且对overlaps置零，不使用IOU

        if not self.config.only_ans_mlm:
            image_feat, image_loc, image_label, masked_label = self.random_region( # mrm之后的内容
                image_feat, image_loc, num_boxes, is_next, overlaps
            ) # image_label代表当前内容是不是应该在model的loss函数里面进行mrm的计算
            # image_label 是1 * num_bbox大小的list
            # tokens = self._tokenizer.add_special_tokens_single_sentence(tokens) # 加上CLS 和 SEP
            # input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)，这个是换过单词之后的句子
        else:
            image_label = [-1] * num_boxes

        # _max_region_num = 101
        mix_num_boxes = min(int(num_boxes), self._max_region_num) # 不能超过最多bb数量
        mix_boxes_pad = np.zeros((self._max_region_num, 5)) # 101 * 5，5来自于box是5维向量
        mix_features_pad = np.zeros((self._max_region_num, 2048)) # 2048是因为抽取的特征是2048维向量
        mix_features_pad_ori = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes)) # 非padding区域是1，padding区域是0，用以区分
        while len(image_mask) < self._max_region_num:
            image_mask.append(0) # 这个mask是按照bb个数算的，一个数顶一个region，代表这个region应不应该mask

        # added by myslef
        image_label = image_label[:mix_num_boxes]
        while len(image_label) < self._max_region_num:
            image_label.append(-1)
        image_label = torch.from_numpy(np.array(image_label))
        # # added by myself
        # image_label = torch.from_numpy(np.array(entry["image_label"]))
        # entry["image_label"] = image_label

        # shuffle the image location here.
        # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
        # img_idx.append(0)
        # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
        # mix_features_pad[:mix_num_boxes] = features[img_idx]

        # padding，直至填满最大的region数量为止，就是在padding，在padding 0
        mix_boxes_pad[:mix_num_boxes] = image_loc[:mix_num_boxes] # 把有用内容装进来，多余的就是padding的
        mix_features_pad[:mix_num_boxes] = image_feat[:mix_num_boxes] # # 把有用内容装进来，多余的就是padding的
        mix_features_pad_ori[:mix_num_boxes] = image_feat_ori[:mix_num_boxes] # 留住之前的内容

        # tensor化
        features = torch.tensor(mix_features_pad).float()
        features_ori = torch.tensor(mix_features_pad_ori).float()
        image_mask = torch.tensor(image_mask).long() # 有用信息对应mask = 1，padding的mask = 0
        spatials = torch.tensor(mix_boxes_pad).float() # location信息，也是空间信息，spatial

        # entry['image_label'] = image_label # image_label的tensor在下面tensorlize里面

        question = entry["q_token"] # 这里的question就是问题加答案的内容
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]
        lm_label_ids = entry["lm_label_ids"]

        image_target = features_ori

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length)) # 文本和图像互信息，101 * 23

        return (
            features,
            spatials,
            image_mask,
            question,
            lm_label_ids,
            input_mask,
            segment_ids,
            co_attention_mask,
            is_next,
            image_label,
            image_target,
        )

    def __len__(self):
        return len(self.entries) # 这个就是数据集数据的个数

    def random_cap(self, caption): # NSP生成任务
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0

        if self.objective != 2 and random.random() > 0.5: # 不match是1
            caption = self.get_random_caption()
            label = 1
        else:
            label = 0 # match是0

        return caption, label

    def get_random_caption(self): # 随便抓一个caption
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        # caption = self.captions[rand_doc_idx]
        caption = self.entries[rand_doc_idx]["question"] # obtain the text (Q + A)，是文本！

        return caption

    def random_cap_sep(self, caption, answer): # NSP生成任务
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, answer, 0

        if self.objective != 2 and random.random() > 0.5: # 不match是1
            caption, answer = self.get_random_caption_sep()
            label = 1
        else:
            label = 0 # match是0

        return caption, answer, label

    def get_random_caption_sep(self): # 随便抓一个caption
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        # caption = self.captions[rand_doc_idx]
        caption = self.entries[rand_doc_idx]["question"] # obtain the question exclude answer
        answer = self.entries[rand_doc_idx]["answer_text"] # obtain answer text

        return caption, answer

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()
    # tokenizer就是分词器
    def random_word(self, tokens, tokenizer, is_next): # MLM任务
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # not sample mask
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) # 这就是[MASK] special token，是数字

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token) # token应该是一个数值，tokenizer应该是词汇
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1) # 表明是否有mask

        return tokens, output_label # output_label和tokens一样长

    def random_word_only_ans_mlm(self, tokens, tokenizer, is_next): # * MLM任务,此处只针对于答案本身进行MLM...
        # ! 注意的是，在判定token是否等于1时，是以token为中心而不是以答案为中心，因为一个word可能对应两个token
        # ! 比如doing，可能会拆分成do和ing... 目前只mask一个token,在一定mask里面，也会保留原本的noise和原文本的设定...
        # ! 生成0-9之间的随机数 random.randint(0,9),! 千万注意，这里包含9！！！！！！！！！！！！！！！！！！！！！！
        output_label = [] # * -1在后面计算loss的时候会被忽略掉...
        masked_idx = random.randint(0, len(tokens) - 1) # random.randint(0,0) = 0，因为目前只mask一个token，所以放在这里...

        for i, token in enumerate(tokens):
            prob = random.random()
            if i == masked_idx:
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) # 这就是[MASK] special token，是数字

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token) # token应该是一个数值，tokenizer应该是词汇
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1) # 表明是否有mask

        return tokens, output_label # output_label和tokens一样长

    def random_region(self, image_feat, image_loc, num_boxes, is_next, overlaps):
        """
        image_feat is (num_bbox + 1) * 2048, i.e. 101 * 2048
        num_boxes: 101
        """
        # ! 我居然没有mask掉更多的iou更大的区域...
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]) - 1) # N + 1 bbox
        output_label.append(-1) # for first special token

        for i in range(1, num_boxes):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 90% to zero the image region, 10% unaltered
                if prob < 0.9:
                    image_feat[i] = 0 # mask掉相关内容，这样的语法是正确的，可以将整一个example进行mask
                    # mask the overlap regions into zeros，overlaps是N * N，overlaps[i]是 1 * N
                    # masked-label一直在for循环里面做or运算，可以一直保留住N个bbox中究竟哪些需要被mask掉
                    masked_label = np.logical_or(masked_label, overlaps[i - 1] > 0.4) # overlap太大，也需遮盖

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        # 将masked-label中的信息融入到output-label之中，masked label等于1，那么output-label等于1
        masked_label = list(map(int, masked_label))
        for i in range(len(masked_label)):
            if masked_label[i] == 1 and output_label[i + 1] == -1:
                output_label[i + 1] = 1

        return image_feat, image_loc, output_label, masked_label # output label就是image label
