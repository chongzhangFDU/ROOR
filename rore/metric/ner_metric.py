import os
import torch
import logging
import numpy as np
from torchmetrics import Metric


logger = logging.getLogger('lightning')

def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r


class NERMetric(Metric):
    def __init__(self, ner_labels, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.ner_labels = ['O']
        for l in ner_labels:
            self.ner_labels.append(f'B-{l}')
            self.ner_labels.append(f'I-{l}')
        self.reset()

    def decode(self, pred):
        pred.append(0) # 保证成功匹配最后一个实体之后终止
        entities = []
        curr_cls, curr_start = None, None
        for i in range(len(pred)):
            if self.ner_labels[pred[i]].startswith('B-'):
                cls_ = self.ner_labels[pred[i]][2:]
                if curr_cls is None:
                    curr_cls, curr_start = cls_, i
                else:
                    entities.append(f'{curr_cls}-{curr_start}-{i}')
                    curr_cls, curr_start = cls_, i
            elif self.ner_labels[pred[i]].startswith('I-'):
                cls_ = self.ner_labels[pred[i]][2:]
                if curr_cls is None:
                    curr_cls, curr_start = cls_, i
                elif cls_ != curr_cls:
                    entities.append(f'{curr_cls}-{curr_start}-{i}')
                    curr_cls, curr_start = cls_, i
                else: pass
            else:
                if curr_cls is not None:
                    entities.append(f'{curr_cls}-{curr_start}-{i}')
                curr_cls, curr_start = None, None
        return entities


    def update(self, predictions, labels, ori_lengths):
        for pred, label, ori_length in zip(predictions, labels, ori_lengths):
            # 去掉CLS token和PAD token对应位置输出
            pred = pred[1:ori_length].tolist()
            label = label[1:ori_length].tolist()
            # 从模型输出解码出实体
            pred_set = set(self.decode(pred))
            gt_set = set(self.decode(label))
            # 更新预测结果统计
            self.p += len(pred_set)
            self.r += len(gt_set)
            self.c += len(pred_set & gt_set)
            self.cnt += 1


    def compute(self):
        e_f1, e_p, e_r = cal_f1(self.c, self.p, self.r)
        metric = {"f1": e_f1,
                  "precision": e_p,
                  "recall": e_r,
                  "samples": self.cnt,
                  }
        return metric

    def reset(self):
        self.add_state("p", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("r", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("c", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0.), dist_reduce_fx="sum")

