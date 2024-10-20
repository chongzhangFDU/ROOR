import os
import numpy as np
import torch
import logging
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


class REMetric(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.reset()

    def update(self, logits, labels):
        preds = np.argmax(logits, axis=1)
        assert preds.shape == labels.shape
        bs, max_rel = preds.shape
        p, r, c, cnt = 0, 0, 0, 0
        for i in range(bs):
            for j in range(max_rel):
                if labels[i][j] >= 0:
                    cnt += 1
                    if preds[i][j] == 1:
                        p += 1
                        if labels[i][j] == 1: c += 1
                    if labels[i][j] == 1: r += 1
        self.p += p
        self.r += r
        self.c += c
        self.cnt += cnt

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
