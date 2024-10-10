import os
import torch
import logging
from torchmetrics import Metric
import textdistance


logger = logging.getLogger('lightning')


def anls(pred, gold, tau=0.5):
    dis = textdistance.levenshtein.distance(pred.lower(), gold.lower())
    max_len = max(len(pred), len(gold))
    if max_len == 0:
        return 0
    else:
        nl = dis / max_len
        return 1-nl if nl < tau else 0


class ANLSMetric(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.add_state('total_anls', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('cnt', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, pred, golds):
        best_anls = 0
        pred = pred.replace('<pad>', '')
        for gold in golds:
            best_anls = max(best_anls, anls(pred, gold))
        self.total_anls += best_anls
        self.cnt += 1

    def compute(self):
        return {'avg_anls': self.total_anls / self.cnt, 'num_samples': self.cnt}

    def reset(self):
        self.__init__(self.compute_on_step, self.dist_sync_on_step)

