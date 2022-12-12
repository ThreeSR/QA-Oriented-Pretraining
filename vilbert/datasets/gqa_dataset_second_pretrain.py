# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import _pickle as cPickle
import json
import logging
import random

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers.tokenization_bert import BertTokenizer

from ._image_features_reader import ImageFeaturesH5Reader

import h5py

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(args, item, label2ans):
    scores = sorted(item['scores'], reverse=True)
    element = scores[0]
    idx = item['scores'].index(element)
    label = item['labels'][idx]
    answer_text = label2ans[label]

    entry = {
        "question_id": int(item["question_id"]),
        "image_id": item["image_id"],
        "question": item["question"],
        "answer": item,
    }

    if not args.sep:
        entry['question'] = entry['question'] + ' ' + answer_text
    else:
        entry["answer_text"] = answer_text

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

def _load_dataset(args, dataroot, name, clean_datasets, label2ans):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'test'
    """
    if name == "train" or name == "val":
        items_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
    elif name == "trainval": # here
        items_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
        items = items[:-3000]
    elif name == "minval": # here
        items_path = os.path.join(dataroot, "cache", "trainval_target.pkl")
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
        items = items[-3000:] # reserve 3000 data points for val
    elif name == "test":
        items_path = os.path.join(dataroot, "testdev_balanced_questions.json")
        items = json.load(open(items_path, "rb"))
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for item in items:
            it = items[item]
            entry = {
                "question_id": int(item),
                "image_id": it["imageId"],
                "question": it["question"],
            }
            entries.append(entry)
    else:
        entries = []
        remove_ids = []
        if clean_datasets:
            remove_ids = np.load(os.path.join(dataroot, "cache", "genome_test_ids.npy"))
            remove_ids = [int(x) for x in remove_ids]
        for item in tqdm(items, desc="Create Entry"):
            if "train" in name and int(item["image_id"]) in remove_ids:
                continue
            if len(item['scores']) == 0: continue
            entries.append(_create_entry(args, item, label2ans))
    return entries


class GQALoader(Dataset):
    def __init__(
        self,
        args,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        clean_datasets,
        padding_index: int = 0,
        max_seq_length: int = 16, # 26
        max_region_num: int = 37, # 101
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        # extract image features (only objects)
        # img_info_path = os.path.join(dataroot, "cache", "gqa_objects_features.pkl")
        # self.img_info = cPickle.load(open(img_info_path, "rb"))
        gqa_objects_info_path = '/dvmm-filer3a/users/rui/multi-task/datasets/gqa/gqa_objects/objects/gqa_objects_info.json'
        with open(gqa_objects_info_path, 'rb') as f:
            self.obj_info = json.load(f)

        self.visualization = False
        self.objective = 1 # 这个需要根据pretrain_second文件里面的配置进行更改...

        clean_train = "_cleaned" if clean_datasets else ""

        if "roberta" in bert_model:
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
                task + "_" + split + "_" + str(max_seq_length) + clean_train + "_qa" +".pkl",
            )
            # cache_path = '/dvmm-filer3a/users/rui/multi-task/datasets/gqa/cache/test.pkl' # for test

        if args.sep: # if use sep token
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + clean_train + '_qa' + "_sep" + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.entries = _load_dataset(args, dataroot, split, clean_datasets, self.label2ans)
            self.num_caps = len(self.entries) # 得到一共多少question，多少文本
            if args.sep:
                self.tokenize_sep(max_seq_length)
            else:
                self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))
            self.num_caps = len(self.entries) # 得到一共多少question，多少文本

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in tqdm(self.entries, desc="Tokenizes"):
            # tokens = self._tokenizer.tokenize(entry["question"])
            # tokens = ["[CLS]"] + tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in tokens
            # ]
            # ITM
            caption, label = self.random_cap(entry["question"])
            is_next = label
            tokens = self._tokenizer.encode(caption)
            # MLM
            tokens, tokens_label = self.random_word(tokens, self._tokenizer, is_next)
            tokens = tokens[: max_length - 2]
            tokens_label = tokens_label[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)
            lm_label_ids = [-1] + tokens_label + [-1] # 这一部分留存了MLM的正确答案，无关部分都是-1

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                gap = max_length - len(tokens)
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding
                padding2 = [-1] * gap
                lm_label_ids += padding2

            assert_eq(len(tokens), max_length)
            assert_eq(len(lm_label_ids), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids
            entry['is_next'] = is_next # 这个仅仅只是一个数值，用于标记
            entry['lm_label_ids'] = lm_label_ids # 这个之后需要输入模型，这个是一个list，应该进行tensor化

    def tokenize_sep(self, max_length=16): # ! 此处的question只有question，没有answer在后面
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in tqdm(self.entries, desc="Tokenizes_sep"):
            # ITM
            caption, answer, label = self.random_cap_sep(entry["question"], entry["answer_text"])
            is_next = label
            tokens_q = self._tokenizer.encode(caption) # question 编码，这时候从原文本变成token，这个是数字
            tokens_a = self._tokenizer.encode(answer) # encode answer
            len_q = len(tokens_q)
            len_a = len(tokens_a)
            tokens = tokens_q + tokens_a # 整条文本
            # 这里是为了MLM
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

    def tensorize(self):

        for entry in tqdm(self.entries, desc="Tensorize"):
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            lm_label_ids = torch.from_numpy(np.array(entry["lm_label_ids"]))
            entry["lm_label_ids"] = lm_label_ids

            if "test" not in self.split:
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

    def __getitem__(self, index):
        entry = self.entries[index]
        is_next = entry['is_next']
        image_id = entry["image_id"]
        # question_id = entry["question_id"]
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        # key: img_id, values: dict like below
        # dict_keys(['width', 'objectsNum', 'idx', 'height', 'file', 'bboxes', 'features'])
        # info = self.img_info[str(image_id)] # 500 * 333
        h5_path = '/dvmm-filer3a/users/rui/multi-task/datasets/gqa/gqa_objects/objects/gqa_objects_'
        h5_path2 = h5_path + str(self.obj_info[str(image_id)]['file']) + '.h5'

        f2 = h5py.File(h5_path2,'r')

        # features = info['features']
        # num_boxes = info['objectsNum']
        # boxes = info['bboxes']
        # img_w = info['width'] # 500
        # img_h = info['height'] # 333

        features = f2['features'][self.obj_info[str(image_id)]['idx']]
        num_boxes = self.obj_info[str(image_id)]['objectsNum']
        boxes = f2['bboxes'][self.obj_info[str(image_id)]['idx']]
        img_w = self.obj_info[str(image_id)]['width'] # 500
        img_h = self.obj_info[str(image_id)]['height'] # 333

        # ! 归一化操作，还需要加入面积
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0]) / (img_h * img_w)
        ) # 加上面积维度, (x1, y1) & (x2, y2)

        image_location[:, (0, 2)] /= img_w
        image_location[:, (1, 3)] /= img_h

        boxes = image_location

        image_feat_ori = features # 留着之后求特征回归,原本的feature，没有mask
        overlaps = iou(boxes, boxes) # 经过计算，归一化与否应该不影响IOU的计算

        # ! NOTE 这里记得要手动变化是否使用IOU，现在先不使用
        set_zero = np.zeros_like(overlaps)
        overlaps = set_zero # ! 姑且对overlaps置零，不使用IOU

        features, boxes, image_label, masked_label = self.random_region( # mrm之后的内容
            features, boxes, num_boxes, is_next, overlaps
        ) # image_label代表当前内容是不是应该在model的loss函数里面进行mrm的计算

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))
        mix_features_pad_ori = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        image_label = image_label[:mix_num_boxes]
        while len(image_label) < self._max_region_num:
            image_label.append(-1)
        image_label = torch.from_numpy(np.array(image_label))

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
        mix_features_pad_ori[:mix_num_boxes] = image_feat_ori[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        features_ori = torch.tensor(mix_features_pad_ori).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]
        lm_label_ids = entry["lm_label_ids"]
        image_target = features_ori

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

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
        return len(self.entries)

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

    def random_region(self, image_feat, image_loc, num_boxes, is_next, overlaps):
        """
        image_feat is (num_bbox + 1) * 2048, i.e. 101 * 2048
        num_boxes: 101
        """
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