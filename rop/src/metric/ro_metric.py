import logging

import torch
from torch import Tensor
from torchmetrics import Metric


logger = logging.getLogger('lightning')


class PRFA(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('cnt', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, logits, grid_labels, num_units):
        """
        :param logits: (bs, 1, seqlen, seqlen)
        :param grid_labels: (bs, seqlen, seqlen)
        :param num_units: (bs, )
        """
        for logit, label, n in zip(logits, grid_labels, num_units):
            n = n.item()
            logit = logit.squeeze(0)
            logit, label = logit[:n, :n], label[:n, :n]
            pred = torch.where(
                logit > 0, torch.ones_like(logit, dtype=torch.long), torch.zeros_like(logit, dtype=torch.long))
            tfpns = (pred * 2 + label).view(-1) # 每一位有 00 01 10 11 四种情况
            tfpns = torch.bincount(tfpns, minlength=4)
            self.tp += tfpns[3]
            self.fp += tfpns[2]
            self.fn += tfpns[1]
            self.tn += tfpns[0]
            self.cnt += 1

    def compute(self):
        p = self.tp / (self.tp + self.fp)
        r = self.tp / (self.tp + self.fn)
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        a = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

        return {'f1': f, 'precision': p, 'recall': r, 'accuracy': a, 'num_samples': self.cnt}

    def __hash__(self) -> int:
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))
