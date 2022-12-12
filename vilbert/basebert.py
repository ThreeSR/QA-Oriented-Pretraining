# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from vilbert.utils import cached_path
from pytorch_transformers.modeling_bert import BertConfig
import pdb
from torch.nn.utils.weight_norm import weight_norm

# from .file_utils import cached_path
logger = logging.getLogger(__name__)


PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TF_WEIGHTS_NAME = "model.ckpt"


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()

        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )

        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)): # 这两个部分的初始化
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm): # layernorm的初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        *inputs,
        **kwargs
    ):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        CONFIG_NAME = "bert_config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        TF_WEIGHTS_NAME = "model.ckpt"

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file,
                )
            )
            return None

        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info(
                "loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file
                )
            )
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        elif resolved_archive_file[-3:] == "bin":
            serialization_dir = "/".join(resolved_archive_file.split("/")[:-1])
            WEIGHTS_NAME = resolved_archive_file.split("/")[-1]
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info(
                "extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir
                )
            )
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        # config_file = os.path.join(serialization_dir, CONFIG_NAME)
        # config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location="cpu")
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(
            s.startswith("bert.") for s in state_dict.keys()
        ):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=0
        ) # 512
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, padding_idx=0
        ) # 在nn.embedding中，padding_idx指示哪一个数值是padding的，比如这里是0，代表
        # 嵌入层输入的内容中，凡是0，都是padding的结果；如果是idx=3，那么代表内容中凡是3
        # 都是padding的结果
        # nn.embedding中，第一个是字典大小，这个参数不能随意设定，需要根据输入词的字典大小而定
        # 可以把这个过程理解为one-hot的过程，之后降维到embedding-dim上面；所以说embedding是layers
        # 做了线性变换，需要学习dense layer里面做变换的参数

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1) # max length， input ids BS * MAX LEN
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # 这一步先unsqueeze，增加一个batch size的维度
        # 之后expand as，就是复制，将一样的内容复制batch size遍，使得position id的维度和input ids的维度相同
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids) # type比较简单，就是直接复刻，然后全部都是0即可

        words_embeddings = self.word_embeddings(input_ids) # bs * 512 * 768, 但0作为padding内容，并没有进行embedding，被置为0而已
        position_embeddings = self.position_embeddings(position_ids) # bs * 512 * 768
        token_type_embeddings = self.token_type_embeddings(token_type_ids) # bs * 512 * 768
        # 先各自embedding，之后再相加，而不是一开始就相加
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # bs * 512 * 768


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(2048, config.hidden_size) # 注意！！这里是nn.linear！不是nn.embedding，
        # 看来之前对于nn.embedding的理解是对的，两种模态对于embedding的处理不同
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, padding_idx=0
        )
        self.image_location_embeddings = nn.Linear(5, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc, token_type_ids=None): # input_ids就是input_imgs
        seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids) # 这里是有的，是全部为1

        image_embeddings = self.image_embeddings(input_ids) # bs * 36 * 2048  *  2048 * 768
        # position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        loc_embeddings = self.image_location_embeddings(input_loc) # bs * 36 * 5  *  5 * 768

        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = image_embeddings + token_type_embeddings + loc_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # bs * 36 * 768


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size) # 768*768
        self.key = nn.Linear(config.hidden_size, self.all_head_size) # 768*768
        self.value = nn.Linear(config.hidden_size, self.all_head_size) # 768*768

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + ( # bs * len * 768 -> bs * len * 12 * 64
            self.num_attention_heads,
            self.attention_head_size,
        ) # 这个的意思是，把原本的768拆成12*64，也就是在这个地方，把原本的内容拆成不同attention head去处理
        # 至于为什么需要multi-head，原文解释是学习不同特征，关注不同内容；但有不同理解....我们也可以姑且类比于CNN，CNN也用了很多kernel
        x = x.view(*new_x_shape) # 将原本x的shape转换成新的shape
        return x.permute(0, 2, 1, 3) # 这里改变 bs * len * 12 * 64 -> bs * 12 * len * 64，原因很简单，就是把len * 64
    # 分到一个head里面处理，这就是permute的意义

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states) # bs * len * 768
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # bs * 12 * len * 64 也就是下面这一行的注释
        key_layer = self.transpose_for_scores(mixed_key_layer) # (batch_size, num_attention_heads, sequence_length, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # torch.matmul 当输入是都是二维时，就是普通的矩阵乘法
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # 为了相乘，所以key-layer转置了一下
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask # 把没必要关心的内容mask掉，比如padding的内容

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # 计算attention scores

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # soft addressing
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # bs * 12 * len * 64 -> bs * len * 12 * 64，准备变回去
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # 变回去了，bs * len * 12 * 64 到 bs * len * 768
        context_layer = context_layer.view(*new_context_layer_shape) # 改变shape，像之前那样
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 768*768
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # short-cut connection
        return hidden_states


class BertAttention(nn.Module): # 一整个attention模块
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module): # 这里是FFN。维度先变大，之后再变小，可能是为了增加网络的学习能力？？
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # 768 * 3072
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module): # 这里，又把前面增加的维度减少了回去，目的应该是为了方便处理，使得输入输出维度相同，方便叠加多层layers
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size) # 3072 * 768
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module): # 一整个layer
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config) # 这是一个完整的paper中的encoder结构
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )# 12层layer，直接复制

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers # 这里把所有layers都保留下来了，但其实最终只需要用到最后一层的layer


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module): # 常规网络，联想CLS也是被提取，这里做一下head transform也make sense
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 768*768
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act] # 激活函数
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states): # bs * len * 768
        hidden_states = self.dense(hidden_states) # bs * len * 768
        hidden_states = self.transform_act_fn(hidden_states) # bs * len * 768
        hidden_states = self.LayerNorm(hidden_states) # bs * len * 768
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear( #  # bs * len * 768 ？？？
            bert_model_embedding_weights.size(1), # 768
            bert_model_embedding_weights.size(0), # len
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states): # # bs * len * 768
        hidden_states = self.transform(hidden_states) # bs * len * 768
        hidden_states = self.decoder(hidden_states) + self.bias # bs * len * len ？？？？？
        return hidden_states


class BertOnlyMLMHead(nn.Module): # MLM
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output): # 注意！这里是sequence来处理MLM问题，和下面不同！
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module): # NSP
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output): # 需要注意的是，这里需要拿pooled_output来分类
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPredictionHeadTransform(nn.Module): # 怎么和上面一样？？
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module): # 怎么和上面一样？？
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertImagePredictionHead(nn.Module): # MRM
    def __init__(self, config, bert_model_embedding_weights):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), 1601) # 768*1601

    def forward(self, hidden_states): # bs * 36 * 768
        hidden_states = self.transform(hidden_states)  # bs * len * 768
        hidden_states = self.decoder(hidden_states) # bs*len*1601
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights): # bert_model_embedding_weights？？
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights) # 文本预测
        self.seq_relationship = nn.Linear(config.hidden_size, 2) # 文本图像关系预测
        self.imagePredictions = BertImagePredictionHead(
            config, bert_model_embedding_weights
        ) # 图像预测

    def forward(self, sequence_output_t, sequence_output_v, pooled_output): # 这里代码有误，函数前后对应错误！！！！

        img_prediction_scores = self.imagePredictions(
            sequence_output_v
        )  # sequence_output[:,txt_len:]) 后面是图像
        prediction_scores = self.predictions(
            sequence_output_t
        )  # sequence_output[:,:txt_len]) 前面是文本

        seq_relationship_score = self.seq_relationship(pooled_output)
        return img_prediction_scores, prediction_scores, seq_relationship_score


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.image_embeddings = BertImageEmbeddings(config) # 图像embedding
        self.embeddings = BertEmbeddings(config) # 语言的embed
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_txt,
        input_imgs, # img_feat bs * 36 * 2048
        image_loc,
        token_type_ids=None, # segment ids 有的
        attention_mask=None, # 语言输入的mask，这是东西是有的
        image_attention_mask=None, # 图像输入的mask，是有的
        output_all_encoded_layers=True,
    ):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None: # 图像不像是语言，一开始就是独热码的形式
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt) # type_as --按照给定的tensor的数据类型转换数据类型

        image_token_type_ids = torch.ones(
            input_imgs.size(0), input_imgs.size(1)
        ).type_as(token_type_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        img_embeding_output = self.image_embeddings(
            input_imgs, image_loc, image_token_type_ids
        ) # bs*32*768

        embedding_output = self.embeddings(input_txt, token_type_ids) # 语言embedding bs*512*768

        embedding_output = torch.cat([embedding_output, img_embeding_output], dim=1) # 两个embedding融合
        # bs*（32 + 512）*768
        extended_attention_mask = torch.cat(
            [extended_attention_mask, extended_image_attention_mask], dim=3 # 两个attention融合
        )

        encoded_layers = self.encoder( # 这里指的是全部的12个（base）layers的输出
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) # 这里抽的就是最后一层第一个cls的内容
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMultiModalPreTraining(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked multi modal modeling head, and
        - the image caption classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMultiModalPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.apply(self.init_bert_weights)

        self.vis_criterion = nn.KLDivLoss(reduction="none") # KL在计算图像的feature差异，reduction="none"代表之后要手动计算最终的kl
        self.loss_fct = CrossEntropyLoss(ignore_index=-1) # -1的index，不用于计算loss，这一点和前面的mask吻合

    def forward( # 这里应该是train函数调用的时候出了问题
        self,
        input_ids,
        image_feat,
        image_target, # 这里是代码有误，对应错误
        image_loc,
        token_type_ids=None, # segment_ids
        attention_mask=None, # input_mask 这个是语言在取相同长度之后的mask，1代表有用，0是padding
        image_attention_mask=None, # image_mask 这个是图像的mask
        masked_lm_labels=None, # -1是非答案部分，会被忽略；有答案的部分是wordpiece之后的token id
        image_label=None, # 表明是否被mask，1代表被mask，0代表没有被mask
        next_sentence_label=None,
    ):
        # in this model, we first embed the images.

        sequence_output, pooled_output = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False, # 这里只输出最后一层的内容
        )
        sequence_output_v = sequence_output[:, input_ids.size(1) :]
        sequence_output_t = sequence_output[:, : input_ids.size(1)]

        prediction_scores_v, prediction_scores_t, seq_relationship_score = self.cls(
            sequence_output_t, sequence_output_v, pooled_output
        )

        # prediction_scores_v, prediction_scores_t, seq_relationship_score = self.cls(
        #     sequence_output, pooled_output, input_ids.size(1), image_feat.size(1)
        # )

        if masked_lm_labels is not None and next_sentence_label is not None:

            prediction_scores_v = prediction_scores_v[:, 1:] # ？？

            img_loss = self.vis_criterion( # 这里即使是feature regression，也是使用kl散度计算；如果是soft label distribution，也是kl
                F.log_softmax(prediction_scores_v, dim=2), image_target
            ) # KL损失，所以image target应该是一个distribution，上述求KL散度是在两个feature的基础上求值的
            # dim=2应该是因为batch*数量*每一个的维度；log softmax就是先求softmax变成概率然后再求log，这是
            # 为了符合kl散度的公式；这里target没有计算log，应该是原本的image target就包含了log的计算？？
            masked_img_loss = torch.sum(
                img_loss * (image_label == 1).unsqueeze(2).float()
            ) / max(torch.sum((image_label == 1)), 0) # 这一步在做的事情就是手动计算kl的真实值，也就是batchmean
            # 1代表要计算的loss，-1代表不需要计算的loss

            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size), # vocab_size = 30522
                masked_lm_labels.view(-1), # view（-1），不管是什么维度的tensor，都会变成相应长度的列向量或行向量
            ) # masked_lm_labels用来指示正确答案的index，prediction_scores_t就是整个答案的预测分布
            # masked_lm_labels和prediction_scores_t的维度可以不相同，masked_lm_labels只是起到指示作用
            # 但prediction_scores_t需要变成可以被指示的维度形式

            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            ) # 这是一个二分类问题，答案是0或者1；所以next_sentence_label是0或者1，没有-1，这里不牵扯mask的问题

            return masked_lm_loss, masked_img_loss, next_sentence_loss
        else:
            return prediction_scores, seq_relationship_score, prediction_scores_v


class BaseBertForVLTasks(BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_prob=0.1, default_gpu=True):
        super(BaseBertForVLTasks, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.vil_prediction = SimpleClassifier(
            config.hidden_size, config.hidden_size * 2, num_labels, 0.5
        )
        # self.vil_prediction = nn.Linear(config.bi_hidden_size, num_labels)
        self.vil_logit = nn.Linear(config.hidden_size, 1)
        self.vision_logit = nn.Linear(config.hidden_size, 1)
        self.linguisic_logit = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
    ):
        sequence_output, pooled_output = self.bert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )

        sequence_output_v = sequence_output[:, input_txt.size(1) :]
        sequence_output_t = sequence_output[:, : input_txt.size(1)]

        vil_prediction = 0
        vil_logit = 0
        vil_binary_prediction = 0
        vision_prediction = 0
        vision_logit = 0
        linguisic_prediction = 0
        linguisic_logit = 0

        vision_prediction, linguisic_prediction, vil_binary_prediction = self.cls( # 此处代码没错
            sequence_output_t, sequence_output_v, pooled_output
        )

        vil_prediction = self.vil_prediction(pooled_output)
        vil_logit = self.vil_logit(pooled_output)
        vision_logit = self.vision_logit(self.dropout(sequence_output_v)) + (
            (1.0 - image_attention_mask) * -10000.0
        ).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        linguisic_logit = self.linguisic_logit(self.dropout(sequence_output_t))

        return (
            vil_prediction,
            vil_logit,
            vil_binary_prediction,
            vision_prediction,
            vision_logit,
            linguisic_prediction,
            linguisic_logit,
        )


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
