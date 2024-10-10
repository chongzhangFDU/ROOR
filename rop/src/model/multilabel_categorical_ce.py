import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)


def multilabel_categorical_crossentropy(y_pred, y_true, mask):
    """
    https://kexue.fm/archives/7359
    多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
         mask和y_true的shape一致，也是非0即1，0表示对应位置不计算loss。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - (y_true + 1-mask) * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1-y_true + 1-mask) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def globalpointer_loss(y_pred, y_true, mask=None):
    """
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    mask:(batch_size, seq_len) or None
    """
    batch_size, ent_type_size, seq_len, seq_len_ = y_pred.shape
    assert seq_len == seq_len_
    if mask is None:
        mask = torch.ones((batch_size, ent_type_size, seq_len, seq_len)).long()
    else:
        mask = torch.einsum('ij,ik->ijk', mask, mask)
        mask = mask.repeat(1, ent_type_size, 1, 1)
    mask = mask.to(y_pred.device)
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    mask = mask.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_pred, y_true, mask)
    return loss


def sparse_multilabel_categorical_crossentropy(
    y_true, y_pred, mask_zero=False, epsilon=1e-7, Inf=1e12
):
    """稀疏版多标签分类的交叉熵
    https://github.com/JunnYu/GPLinker_pytorch/blob/dev/models.py
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + Inf
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=epsilon, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def sparse_globalpointer_loss(y_true, y_pred):
    shape = y_pred.shape
    # bs, nclass, max_spo_num
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # bs, nclass, seqlen * seqlen
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    loss = sparse_multilabel_categorical_crossentropy(
        y_true, y_pred, mask_zero=True)
    return loss.sum(dim=1).mean()