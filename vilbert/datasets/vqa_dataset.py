# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected): # 自己写的一个判断是否相等的函数
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer, label2ans):
    # 原本answers：{'image_id': 458752, 'labels': [1164], 'scores': [1], 'question_id': 458752000}
    # ! 这里的scores就是vqa scores，作者已经做过了预处理
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

    # * FIXME *#
    scores = sorted(answer['scores'], reverse=True)
    element = scores[0]
    idx = answer['scores'].index(element)
    label = answer['labels'][idx]
    answer_text = label2ans[label]
    entry["question"] = answer_text
    # * FIXME *#

    return entry


def _load_dataset(dataroot, name, clean_datasets, label2ans): # 基本上都是读取问题，读取答案，进行排序，进行match
    # # dataroot /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
    # name trainval; clean_datasets = _cleaned
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
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

    elif name == "trainval": # here we go :)
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

    elif name == "minval":
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
        questions = questions_val[-3000:]
        answers = answers_val[-3000:]

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
        for question, answer in zip(questions, answers):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue # 这部分内容删去，不使用。这就是clean掉的部分，所以cleaned用意在此
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])

            # * FIXME *#
            if len(answer['scores']) == 0: continue # 这一行代码的目的是滤除没有答案的问题，这部分占比很小，大约2%
            # * FIXME *#

            entries.append(_create_entry(question, answer, label2ans)) # 这就是留下的，最后使用的数据部分

    return entries # 字典，里面存着问答内容，图像id，问题id。简单来说就是question id + V + Q + A
            # 注意！！：这里的entries是还没有token的内容，就是原始文本


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        task, # VQA
        dataroot, # /dvmm-filer3a/users/rui/multi-task/datasets/VQA/
        annotations_jsonpath, # 没用上...
        split, # trainval
        image_features_reader, # /dvmm-filer3a/users/rui/multi-task/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb
        gt_image_features_reader,
        tokenizer, # BertTokenizer.from_pretrained，分词器
        bert_model, # default="bert-base-uncased"
        clean_datasets, # default=True
        padding_index=0,
        max_seq_length=16, # 这个应该还是23
        max_region_num=100, # ! 因为改了图像读取器，所以这里要变成 36 + 1 = 37 先36，有一个处理的地方找不到了...
    ):
        super().__init__()
        self.split = split # trainval
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
        imgid2img_trainval_path = os.path.join(dataroot, "cache", "Imgid2img_trainval2014_100.pkl") # !! 注意这里是36还是100
        self.imgid2img_trainval = cPickle.load(open(imgid2img_trainval_path, "rb"))

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
                # task + "_" + split + "_" + str(max_seq_length) + clean_train + "_qa" + "_sep" +".pkl", # !!!!!!!!!
                # task + "_" + split + "_" + str(max_seq_length) + clean_train + "_qa" + "_sep" + "_id" +".pkl", # !!!!!!!!!
                task + "_" + split + "_" + str(max_seq_length) + clean_train +".pkl", # !!!!!!!!!
                # task + "_" + split + "_" + str(max_seq_length) + clean_train + "_answer" + ".pkl", # !!!!!!!!!
            ) # 存储已经toeknizer好的数据文件
            # cache_path = /dvmm-filer3a/users/rui/multi-task/datasets/VQA/cache/VQA_trainval_23_cleaned.pkl

        if not os.path.exists(cache_path): # 这边的操作就是，把上面的原始文本token化
            self.entries = _load_dataset(dataroot, split, clean_datasets, self.label2ans) # 这就是前面的函数，得到ID + VQA
            self.tokenize(max_seq_length) # 23，调用了下面的函数
            self.tensorize() # tensor化，所以这个数据集之后需要在torch环境下才可以打开
            cPickle.dump(self.entries, open(cache_path, "wb")) # 存储已经处理好的数据为pkl文件
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb")) # 如果已经有处理好的文件，那么直接读取

    def tokenize(self, max_length=16): # max_length是怎么确定的？目前max length是23
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset. 在原本数据集上增加token之后的question内容
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"]) # question编码，这时候从原文本变成token，这个是数字，encode一步
            # 到位，encode实现了先toknize再变成数字的步骤
            tokens = tokens[: max_length - 2] # 空出位置
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens) # 加入cls之类的特殊token，前面cls，后面sep

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens)) # 这个_padding_index就是pad谁，这里是0
                tokens = tokens + padding
                input_mask += padding # input_mask应该就是表明哪里是文字，文字对应1，哪里是padding，padding对应0
                segment_ids += padding

            assert_eq(len(tokens), max_length) # 确认上述操作
            # 在原本的entry中加入新的内容，加上token之后的成果
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self): # 将变量tensor化

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

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
    def __getitem__(self, index): # 在原本只有处理好的文本基础上，读取图像feature
        entry = self.entries[index] # 获取某个entry
        image_id = entry["image_id"] # 得到两个id
        question_id = entry["question_id"]
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id] # 根据image id得到feature
        # eg: array..., 101, boxes。第四个不用管，是一个什么original...
        # 第一个array是各个box的features，所以是num_bbox * 2048维，比如101个bbox，每个bbox的特征是2048维向量
        # 第三个boxes是坐标位置信息
        # _max_region_num = 101

        # 如果使用10-100，以下面为准，如果使用36，以更下面为准
        # FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

        img = self.imgid2img_trainval[str(image_id).zfill(12)]
        # imgid2img_trainval['000000573195']
        # id都是唯一的，和train与val无关；格式如上，之后得到具体字典，字典的keys如下
        # odict_keys(['img_id', 'img_h', 'img_w', 'objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'num_boxes', 'boxes', 'features'])
        num_boxes = img['num_boxes']
        features = img['features']
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

        mix_num_boxes = min(int(num_boxes), self._max_region_num) # 不能超过最多bb数量
        mix_boxes_pad = np.zeros((self._max_region_num, 5)) # 101 * 5，5来自于box是5维向量

        mix_features_pad = np.zeros((self._max_region_num, 2048)) # 2048是因为抽取的特征是2048维向量

        image_mask = [1] * (int(mix_num_boxes)) # 非padding区域是1，padding区域是0，用以区分
        while len(image_mask) < self._max_region_num:
            image_mask.append(0) # 这个mask是按照bb个数算的，一个数顶一个region，代表这个region应不应该mask

        # shuffle the image location here.
        # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
        # img_idx.append(0)
        # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
        # mix_features_pad[:mix_num_boxes] = features[img_idx]

        # padding，直至填满最大的region数量为止，就是在padding，在padding 0
        mix_boxes_pad[:mix_num_boxes] = image_location[:mix_num_boxes] # 把有用内容装进来，多余的就是padding的
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes] # # 把有用内容装进来，多余的就是padding的
        # tensor化
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long() # 有用信息对应mask = 1，padding的mask = 0
        spatials = torch.tensor(mix_boxes_pad).float() # location信息，也是空间信息，spatial

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length)) # 文本和图像互信息，101 * 23
        target = torch.zeros(self.num_labels) # 3129个0

        if "test" not in self.split: # ：） go go go
            answer = entry["answer"] # {'labels': [1164], 'scores': [1]}
            labels = answer["labels"] # [1164]，已经tensor化！
            scores = answer["scores"] # [1]
            if labels is not None:
                target.scatter_(0, labels, scores) # 这个函数在这里的意思是，将label对应的score送到target上
            # eg：target = torch.zeros(3129)
                #   labels = torch.tensor([1, 2])
                #   scores = torch.tensor([1, 0.6])
                #   target.scatter_(0, labels, scores)
                #   print(target)
                #   tensor([0.0000, 1.0000, 0.6000,  ..., 0.0000, 0.0000, 0.0000])

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        )

    def __len__(self):
        return len(self.entries)
