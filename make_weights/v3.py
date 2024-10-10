import torch


def expand_v3(ori_path, new_path):
    st_dict = torch.load(ori_path)
    st_dict['layoutlmv3.embeddings.position_ids'] = \
        torch.cat(
            (st_dict['layoutlmv3.embeddings.position_ids'],
             st_dict['layoutlmv3.embeddings.position_ids'],
             st_dict['layoutlmv3.embeddings.position_ids'],
             st_dict['layoutlmv3.embeddings.position_ids'],), dim=1)
    st_dict['layoutlmv3.embeddings.position_embeddings.weight'] = \
        torch.cat(
            (st_dict['layoutlmv3.embeddings.position_embeddings.weight'],
             st_dict['layoutlmv3.embeddings.position_embeddings.weight'],
             st_dict['layoutlmv3.embeddings.position_embeddings.weight'],
             st_dict['layoutlmv3.embeddings.position_embeddings.weight'],), dim=0)
    torch.save(st_dict, new_path)
    
    
expand_v3(
    '/path/to/layoutlmv3-base/pytorch_model.bin',
    '/path/to/layoutlmv3-base-2048/pytorch_model.bin')
expand_v3(
    '/path/to/layoutlmv3-large/pytorch_model.bin',
    '/path/to/layoutlmv3-large-2048/pytorch_model.bin')