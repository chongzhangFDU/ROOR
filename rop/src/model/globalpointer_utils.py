import torch


def add_sequence_masking(x, mask, axis=None):
    if mask is None:
        return None
    else:
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        # 使用broadcast_to则无法导出onnx
        # return 1 - torch.broadcast_to(mask, x.shape)
        return torch.ones_like(x, dtype=mask.dtype) - mask


def add_tril_mask(x, diagonal=0):
    l = x.shape[-1]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(x.shape)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask <= arange
    return mask


def add_neg_mask(logits, mask, neg_mask_rate=0.0):
    # 负样本采样
    # 输入mask为 int, 需要转换成bool，最终输出也为int
    probability_matrix = torch.full(logits.shape, neg_mask_rate).type_as(logits)
    # probability_matrix.masked_fill_(mask, value=0.0)
    probability_matrix.masked_fill_(mask.bool(), value=0.0)
    neg_mask = torch.bernoulli(probability_matrix)
    # final_mask = torch.logical_or(mask, neg_mask)
    final_mask = mask + neg_mask.int()
    return final_mask


def add_neg_mask_v2(logits, mask, neg_mask_rate=0.0):
    # 负样本采样
    # 输入mask为 int, 需要转换成bool，最终输出也为int
    probability_matrix = torch.full(logits.shape, neg_mask_rate).type_as(logits)
    # probability_matrix.masked_fill_(mask, value=0.0)
    probability_matrix.masked_fill_(mask.bool(), value=0.0)
    neg_mask = torch.bernoulli(probability_matrix)
    # final_mask = torch.logical_or(mask, neg_mask)
    return neg_mask.int()


def add_pad_tril_neg_mask(logits, mask, neg_mask_rate=0.0):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    pad_mask1 = add_sequence_masking(logits, mask, axis=logits.ndim - 2)
    pad_mask2 = add_sequence_masking(logits, mask, axis=logits.ndim - 1)
    tril_mask = add_tril_mask(torch.ones_like(logits), diagonal=-1)
    # final_mask = torch.logical_or(torch.logical_or(pad_mask1, pad_mask2), tril_mask)
    # final_mask is int type
    final_mask = torch.logical_or(pad_mask1, pad_mask2).int() + tril_mask.int()
    if neg_mask_rate > 0.0:
        final_mask = add_neg_mask(logits, final_mask, neg_mask_rate)
    return final_mask


def self_sequence_masking(x, mask, value='-inf'):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        return x * (1 - mask) + value * mask


def tril_onnx(x, diagonal=0):
    mask = add_tril_mask(x, diagonal=diagonal)
    return x.masked_fill(mask == 0, 0)


# def add_mask_tril(logits, mask):
#     if mask.dtype != logits.dtype:
#         mask = mask.type(logits.dtype)
#
#     logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
#     logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
#     # 排除下三角，如果使用torch.tril，则导出onnx opet=12 会报错
#     # mask = torch.tril(torch.ones_like(logits), diagonal=-1)
#     mask = tril_onnx(torch.ones_like(logits), diagonal=-1)
#     logits = logits - mask * 1e12
#     return logits

#
def add_mask_tril(logits, mask, neg_mask_rate=0.0):
    mask = add_pad_tril_neg_mask(logits, mask, neg_mask_rate=neg_mask_rate)
    # logits = self_sequence_masking(logits, mask.int(), '-inf')
    logits = self_sequence_masking(logits, mask, '-inf')
    # mask is bool
    return logits, mask


def lengths_to_mask(seq_len, max_len=None):
    r"""
    .. code-block::
        >>> seq_len = torch.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])
    :param torch.LongTensor seq_len: (B,)
    :param int max_len: max sequence length。
    :return:  torch.Tensor  (B, max_len)
    """
    assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
    batch_size = seq_len.size(0)
    max_len = int(max_len) if max_len else seq_len.max().long()
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    return mask