import torch
import pickle
import torch.nn.functional as F


# 字符串列表和tensor互转

MAX_NUM_STRINGS = 16
MAX_STRINGS_TOTAL_LENGTH = 512

def strings_to_tensor(strings):
    """
    :param strings: 长度不超过MAX_NUM_STRINGS的字符串列表
    :return bytes: (MAX_STRINGS_TOTAL_LENGTH,) 元数据
    :return lengths: (MAX_NUM_STRINGS,) 每个元素的对应长度
    :return num_strings: (1,) len(strings)
    """

    # assert len(strings) <= MAX_NUM_STRINGS
    strings = strings[:MAX_NUM_STRINGS]
    
    # 使用pickle序列化字符串列表
    serialized_strings = [pickle.dumps(s) for s in strings]
    # 获取每个序列化后的字节长度，并保存为tensor，但总长度不超过MAX_STRINGS_TOTAL_LENGTH，否则截取
    serialized_strings_filtered, lengths = [], []
    num_strings, curr_length = 0, 0
    for s in serialized_strings:
        if curr_length + len(s) <= MAX_STRINGS_TOTAL_LENGTH:
            serialized_strings_filtered.append(s)
            lengths.append(len(s))
            num_strings += 1
            curr_length += len(s)
    lengths += [-1] * (MAX_NUM_STRINGS - num_strings)
    lengths = torch.tensor(lengths)
    # 将所有字节串拼接成一个字节串，转换为PyTorch tensor
    concatenated_bytes = b''.join(serialized_strings_filtered)
    tensor = torch.tensor(list(concatenated_bytes), dtype=torch.uint8)
    assert tensor.shape[0] <= MAX_STRINGS_TOTAL_LENGTH
    tensor = F.pad(tensor, pad=(0, MAX_STRINGS_TOTAL_LENGTH-tensor.shape[0]), mode='constant', value=0)
    # return
    return tensor, lengths, torch.tensor([num_strings])


def tensor_to_strings(tensor, lengths, num_strings):
    """
    :param tensor: PyTorch Tensor
    :param lengths: 编码后的字符串长度列表
    :param num_strings: 答案个数
    :return: 字符串列表
    """
    # 将 tensor 转换回字节串
    concatenated_bytes = bytes(tensor.tolist())

    # 按长度切分字节串并反序列化
    deserialized_strings = []
    start = 0
    for i in range(num_strings):
        deserialized_string = pickle.loads(concatenated_bytes[start:start + lengths[i]])
        deserialized_strings.append(deserialized_string)
        start += lengths[i]

    return deserialized_strings


if __name__ == '__main__':
    tensor, lengths, num_strings = strings_to_tensor(['hello', 'world', 'pytorch'])
    print(tensor.shape, lengths.shape, num_strings)
    deserialized_strings = tensor_to_strings(tensor, lengths, num_strings)
    print(deserialized_strings)  # 输出: ['hello', 'world', 'pytorch']