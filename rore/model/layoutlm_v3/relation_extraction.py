import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from model.layoutlm_v3.modeling_layoutlmv3 import LayoutLMv3PreTrainedModel, LayoutLMv3Model


class BiaffineAttention(nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class LayoutLMv3ForRelationExtraction(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, lam=None, num_ro_layers=None):
        super().__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config, lam=lam, num_ro_layers=num_ro_layers)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 实体表征
        self.num_labels = config.num_labels  # 实体类型数
        self.entity_emb = nn.Embedding(self.num_labels, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        # 关系分类
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

        self.init_weights()

    def entity_embedding(self, sequence_output, entities):
        # sequence_output: (bs, max_len)
        # entities: (bs, max_ent, 4)
        # return: (bs, max_ent, embed_size)
        # semantic embedding
        entity_semantic_emb = self.batch_indexing(sequence_output, entities[:, :, 1])
        # label embedding
        entity_label_idxs = entities[:, :, 3]
        entity_label_emb = self.entity_emb(entity_label_idxs)
        # concat and return
        return torch.cat((entity_semantic_emb, entity_label_emb), dim=-1)

    def batch_indexing(self, embeddings, idxs):
        """ embeddings: (bs, max_len, embed_size)
            idxs: (bs, num_idxs)
            returns: (bs, num_idxs, embed_size)
        """
        idxs = idxs.unsqueeze(-1).expand(-1, -1, embeddings.size(2))
        return embeddings.gather(1, idxs.long())

    def forward(
            self,
            input_ids,
            bbox,
            labels=None,
            images=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            ro_attn=None,
            entities=None,
            entity_pairs=None,
            **kwargs,
    ):
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            images=images,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            ro_attn=ro_attn,
        )

        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        entity_embs = self.entity_embedding(sequence_output, entities) # (bs, max_ent, embed_size)

        head_embs = self.batch_indexing(entity_embs, entity_pairs[:, :, 0]) # (bs, max_rel, embed_size)
        tail_embs = self.batch_indexing(entity_embs, entity_pairs[:, :, 1]) # (bs, max_rel, embed_size)
        heads = self.ffnn_head(head_embs)
        tails = self.ffnn_tail(tail_embs)
        logits = self.rel_classifier(heads, tails).permute(0, 2, 1) # (bs, 2, max_rel)
        loss = self.loss_fct(logits, labels)

        return ModelOutput(
            loss=loss,
            logits=logits,
        )