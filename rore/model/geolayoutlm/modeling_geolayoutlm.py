# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import sys

import torch
from torch import nn
from model.geolayoutlm.bros.configuration_bros import BrosConfig
from model.geolayoutlm.bros.tokenization_bros import BrosTokenizer
from model.geolayoutlm.bros.modeling_bros_convnext import GeoLayoutLMModel, PairGeometricHead, MultiPairsGeometricHead

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GeoLayoutLMVIEModel(nn.Module):
    def __init__(self,
                 config,
                 bert_base_path='bert-base-uncased',
                 model_ckpt_path=None,
                 use_vision=True,
                 linking_coeff=0.5):
        super().__init__()

        self.config = config
        if self.config.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            # backbone
            self.backbone_config = BrosConfig(**self.config.backbone_config)
            self.tokenizer = BrosTokenizer.from_pretrained(
                bert_base_path, do_lower_case=True)
            self.geolayoutlm_model = GeoLayoutLMModel(self.backbone_config)
            # task head
            if self.config.use_inner_id:
                self.bio_classifier = nn.Linear(self.backbone_config.hidden_size * 2, self.config.n_classes)
            else:
                self.bio_classifier = nn.Linear(self.backbone_config.hidden_size, self.config.n_classes)
            self.pair_geometric_head = PairGeometricHead(self.backbone_config)
            self.multi_pairs_geometric_head = MultiPairsGeometricHead(self.backbone_config)
        else:
            raise ValueError(
                f"Not supported model: self.config.backbone={self.config.backbone}"
            )
        self.use_vision = use_vision

        self.dropout = nn.Dropout(0.1)
        self.loss_func_labeling = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_func_linking = nn.BCEWithLogitsLoss(reduction='none')
        self.linking_coeff = linking_coeff

        if model_ckpt_path and self.config.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            self._init_weight(model_ckpt_path)

    def _init_weight(self, model_ckpt_path):
        logger.info("init weight from {}".format(model_ckpt_path))
        state_dict = torch.load(model_ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict_new = dict()
        for key, value in state_dict.items():
            # 对各种权重格式进行兼容
            if 'pair_direct_cls' in key:
                continue
            if key.startswith("ptm_head.predictions.") or key.startswith("ptm_head.triple_geometric_head."):
                continue
            if key.startswith("net."):
                key = key.replace("net.", "")
            elif key.startswith("model."):
                key = key.replace("model.", "")
            if key.startswith("ptm_head."):
                key = key.replace("ptm_head.", "")
            if key in ['bio_classifier.weight', 'bio_classifier.bias']:
                if value.shape[0] != self.config.n_classes:
                    logger.info(f'Ignoring {key}: expect n_classes = {self.config.n_classes}, copying a param with shape {value.shape}')
                    continue
            state_dict_new[key] = value
        result = self.load_state_dict(state_dict_new, strict=False)
        if result.missing_keys:
            logger.info(f"Warning: the following keys are missing in the state_dict: {result.missing_keys}")
        if result.unexpected_keys:
            logger.info(f"Warning: the following keys are unexpected in the state_dict: {result.unexpected_keys}")

    def forward(self, batch):
        """ batch is a dict with the following keys:
        'image', 'input_ids', 'bbox_4p_normalized',
        'attention_mask', 'first_token_idxes', 'block_mask',
        'bbox', 'line_rank_id', 'line_rank_inner_id'
        """

        input_ids = batch["input_ids"]
        image = batch["image"]
        bbox = batch["bbox"]
        bbox_4p_normalized = batch["bbox_4p_normalized"]
        attention_mask = batch["attention_mask"]
        first_token_idxes = batch["first_token_idxes"]
        first_token_idxes_mask = batch["block_mask"]
        line_rank_id = batch["line_rank_id"]
        line_rank_inner_id = batch["line_rank_inner_id"]
        ro_attn = batch["ro_attn"] if "ro_attn" in batch else None

        if self.config.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            # sequence_output: [batch_size, seq_len, hidden_size]
            # blk_vis_features: [batch_size, block_num, hidden_size]
            # text_mm_feat: [batch_size, seq_len, hidden_size]
            # vis_mm_feat: [batch_size, 1+block_num, hidden_size]
            sequence_output, blk_vis_features, text_mm_feat, vis_mm_feat = self.geolayoutlm_model(
                input_ids=input_ids,
                image=image,
                bbox=bbox,
                bbox_4p_normalized=bbox_4p_normalized,
                attention_mask=attention_mask,
                first_token_idxes=first_token_idxes,
                first_token_idxes_mask=first_token_idxes_mask,
                line_rank_id=line_rank_id,
                line_rank_inner_id=line_rank_inner_id,
                ro_attn=ro_attn
            )

        # SER
        if self.config.use_inner_id:
            sequence_output = torch.cat(
                (
                    text_mm_feat,
                    self.geolayoutlm_model.text_encoder.embeddings.line_rank_inner_embeddings(line_rank_inner_id)
                ), 2
            )
            sequence_output = self.dropout(sequence_output)
            logits4labeling = self.bio_classifier(sequence_output)  # [batch_size, seq_len, nc]
        else:
            bio_text_mm_feat = self.dropout(text_mm_feat)
            logits4labeling = self.bio_classifier(bio_text_mm_feat)  # [batch_size, seq_len, nc]

        # RE
        batch_size, blk_len = first_token_idxes.shape
        B_batch_dim = torch.arange(0, batch_size,
                                   device=text_mm_feat.device).reshape(
            batch_size, 1).expand(batch_size, blk_len)

        text_mm_blk_features = text_mm_feat[B_batch_dim, first_token_idxes]
        text_mm_blk_features = text_mm_blk_features * first_token_idxes_mask.unsqueeze(2)

        if self.config.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            visual_mm_blk_features = vis_mm_feat[:,
                                     1:]  # the global image feature; [batch_size, block_num, hidden_size]
            if self.use_vision:
                mixed_blk_features = self.dropout(visual_mm_blk_features + text_mm_blk_features)
            else:
                mixed_blk_features = self.dropout(text_mm_blk_features)

            logits4linking_list = []
            logits4linking = self.pair_geometric_head(mixed_blk_features)  # [batch_size, block_num, block_num]
            logits4linking_list.append(logits4linking)
            logits4linking_ref = self.multi_pairs_geometric_head(mixed_blk_features, logits4linking,
                                                                 first_token_idxes_mask)
            logits4linking_list.append(logits4linking_ref)

        # output and loss
        head_outputs = {
            "logits4labeling": logits4labeling,
            "logits4linking_list": logits4linking_list,
            "max_prob_as_father": self.config.max_prob_as_father,
            "max_prob_as_father_upperbound": self.config.max_prob_as_father_upperbound,
            "is_geo": self.config.backbone in [
                "alibaba-damo/geolayoutlm-base-uncased",
                "alibaba-damo/geolayoutlm-large-uncased",
            ]
        }
        head_outputs["pred4linking"] = torch.where(
            torch.sigmoid(head_outputs["logits4linking_list"][-1]) >= 0.5,
            torch.ones_like(head_outputs["logits4linking_list"][-1]),
            torch.zeros_like(head_outputs["logits4linking_list"][-1]))
        losses = self._get_loss(head_outputs, batch)

        return head_outputs, losses

    def _get_loss(self, head_outputs, batch):
        labeling_loss, linking_loss = 0.0, 0.0
        # labeling loss
        labeling_loss = labeling_loss + self.loss_func_labeling(
            head_outputs["logits4labeling"].transpose(1, 2),
            batch["bio_labels"]
        )
        # linking loss
        for logits_lk in head_outputs["logits4linking_list"]:
            linking_loss_pairwise = self.loss_func_linking(
                logits_lk,
                batch["el_labels_blk"]
            )
            label_mask = batch["el_label_blk_mask"]
            linking_loss_all = torch.mul(linking_loss_pairwise, label_mask)
            linking_loss_all = torch.sum(linking_loss_all) / (label_mask.sum() + 1e-7)

            positive_label_mask = (batch["el_labels_blk"] > 0).float() * label_mask
            linking_loss_positive = torch.mul(linking_loss_pairwise, positive_label_mask)
            linking_loss_positive = torch.sum(linking_loss_positive) / (positive_label_mask.sum() + 1e-7)

            # make each positive prob the same
            prob_lk = torch.sigmoid(logits_lk)
            mu_p = torch.mul(prob_lk, positive_label_mask).sum(2, keepdim=True) / (
                    positive_label_mask.sum(2, keepdim=True) + 1e-7)
            var_p = torch.pow(((prob_lk - mu_p) * positive_label_mask), 2).sum(2) / (
                    positive_label_mask.sum(2) + 1e-7)  # [b, T]
            var_mask = (positive_label_mask.sum(2) > 1).float()
            var_p = (var_p * var_mask).sum(1) / (var_mask.sum(1) + 1e-7)
            var_p = var_p.mean()

            linking_loss = linking_loss + (linking_loss_all + linking_loss_positive + var_p)

        loss_dict = {
            "labeling_loss": labeling_loss,
            "linking_loss": linking_loss,
            "total_loss": (1 - self.linking_coeff) * labeling_loss + self.linking_coeff * linking_loss,
        }
        return loss_dict
