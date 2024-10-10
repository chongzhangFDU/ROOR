import os
import random
import json

import torch
import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

from src.utils.image_utils import RandomResizedCropAndInterpolationWithTwoPic
from src.model.layoutlm_v3.tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast

def points4to2(box):
    '''
    将4点坐标转为2点坐标
    '''
    if box is None or len(box) == 0 or not isinstance(box[0], list):
        return box

    xs, ys = list(zip(*box))
    box = [min(xs), min(ys), max(xs), max(ys)]
    return box


class RoDataset(Dataset):
    def __init__(self,
                 dataset,
                 img_dir,
                 tokenizer,
                 image_input_size=224,
                 image_patch_size=16,
                 max_x=1023,
                 max_y=1023,
                 xy_same_scale=False,
                 box_level='word',
                 unit_type='segment',
                 max_num_units=256,
                 max_seq_len=2048,
                 allow_truncate=True,
                 ):
        super().__init__()

        # 图像输入
        # 图像文件所在目录，结合json obj里面的相对路径，把图读出来
        self.img_dir = img_dir
        # 图像大小，patch大小，patch token长度
        self.image_input_size = image_input_size
        self.image_patch_size = image_patch_size
        self.IMAGE_LEN = int(image_input_size / image_patch_size) ** 2 + 1
        # 增强算法
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
        IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
        imagenet_default_mean_and_std = False
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.common_transform = transforms.Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=image_input_size, interpolation='bicubic'),
        ])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        # 文本输入
        # tokenizer
        self.tokenizer = tokenizer
        # 单条样本text token最大个数
        self.max_seq_len = max_seq_len
        # 推理的时候遇到超长样本，可以允许截断
        self.allow_truncate = allow_truncate 

        # layout输入
        # 使用word level or segment level box
        self.box_level = box_level
        # xy放缩到标准尺寸[0, max_x or max_y)
        self.max_x = max_x
        self.max_y = max_y
        # 是否等比例放缩
        self.xy_same_scale = xy_same_scale

        # 阅读顺序关系预测
        # 对word pair or segment pair进行关系预测
        self.unit_type = unit_type
        # 最多有几个word或segment参与关系对预测
        self.max_num_units = max_num_units

        # dataset: 输入json obj 列表，在建立数据集的时候处理好
        self.dataset = []
        for i in tqdm.tqdm(dataset):
            self.dataset.append(self.process(i))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def process(self, json_obj):
        """ 输入一个符合cdip和ro_linkings标注格式的json_obj
            输出格式参考最后
        """

        # 确定bbox放缩比例
        x_radio = self.max_x / json_obj['img']['width']
        y_radio = self.max_y / json_obj['img']['height']
        if self.xy_same_scale:  # 等比例放缩
            r = min(x_radio, y_radio)
            x_radio, y_radio = r, r

        # 构造input_ids, bboxes, attention_mask
        # 获得input_ids中每个segment/word对应的span，即segment_spans, word_spans
        input_ids = [self.tokenizer.cls_token_id]
        bboxes = [[0, 0, 0, 0]]
        curr_idx = 1
        segment_spans, word_spans = [], []
        for segment in json_obj['document']:
            segment_start_idx = curr_idx
            for word in segment['words']:
                tokens = self.tokenizer(word['text'], add_special_tokens=False).input_ids
                if self.box_level == 'segment':
                    bbox = segment['box']
                elif self.box_level == 'word':
                    bbox = word['box']
                else:
                    raise ValueError(f'self.box_level is {self.box_level}; expected "segment" or "box"')
                bbox = points4to2(bbox)
                bbox = [
                    int(bbox[0] * x_radio), int(bbox[1] * y_radio),
                    int(bbox[2] * x_radio), int(bbox[3] * y_radio)]
                bbox[0] = max(min(bbox[0], self.max_x), 0) # 这是因为有bbox超过height, width范围的情况
                bbox[2] = max(min(bbox[2], self.max_x), 0)
                bbox[1] = max(min(bbox[1], self.max_y), 0)
                bbox[3] = max(min(bbox[3], self.max_y), 0)
                for token in tokens:
                    input_ids.append(token)
                    bboxes.append(bbox)
                word_spans.append((curr_idx, curr_idx + len(tokens)))
                curr_idx += len(tokens)
            segment_spans.append((segment_start_idx, curr_idx))
        input_ids.append(self.tokenizer.sep_token_id)
        bboxes.append([0, 0, 0, 0])
        if self.allow_truncate:
            assert len(input_ids) == len(bboxes) 
            if len(input_ids) > self.max_seq_len:
                # 试图找到这条样本的标识信息，并打印出来
                if 'uid' in json_obj:
                    print(json_obj['uid'])
                elif 'img' in json_obj:
                    print(json_obj['img'])
                print(f'Length exceeds: input tokens {len(input_ids)} > max_seq_len {self.max_seq_len}. Do truncate.')
                # 截断过长部分
                input_ids = input_ids[:self.max_seq_len]
                bboxes = bboxes[:self.max_seq_len]
                word_spans = [(a, b) for (a, b) in word_spans if b <= self.max_seq_len]
                segment_spans = [(a, b) for (a, b) in segment_spans if b <= self.max_seq_len]
        else:
            assert len(input_ids) == len(bboxes) <= self.max_seq_len    
        text_attention_mask = [1 for i in input_ids]
        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.tokenizer.pad_token_id)
            bboxes.append([0, 0, 0, 0])
            text_attention_mask.append(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        bboxes = torch.tensor(bboxes, dtype=torch.long)
        text_attention_mask = torch.tensor(text_attention_mask, dtype=torch.long)

        # 根据segment_spans, word_spans构造unit_mask，指出每一个unit范围
        # 并构造这些unit之间的关系标签grid_label
        if self.unit_type == 'segment':
            # segment id to segment idx
            s_id_to_s_idx = dict()
            for i, segment in enumerate(json_obj['document']):
                s_id_to_s_idx[segment['id']] = i
            # ro标注转为grid label
            if 'ro_linkings' in json_obj:
                grid_labels = [
                    [0 for j in range(self.max_num_units)]
                    for i in range(self.max_num_units)]
                for i, j in json_obj['ro_linkings']:
                    grid_labels[s_id_to_s_idx[i]][s_id_to_s_idx[j]] = 1
                grid_labels = torch.tensor(grid_labels, dtype=torch.long)
            else:
                grid_labels = None
            # 构造每个unit的mask
            unit_masks = []
            for i, j in segment_spans:
                unit_mask = [0] * self.max_seq_len
                unit_mask[i:j] = [1] * (j - i)
                unit_masks.append(unit_mask)
        elif self.unit_type == 'word':
            # segment id to segment idx, word id to word idx
            s_id_to_s_idx = dict()
            w_id_to_w_idx = dict()
            word_idx = 0
            for i, segment in enumerate(json_obj['document']):
                s_id_to_s_idx[segment['id']] = i
                for word in segment['words']:
                    w_id_to_w_idx[word['id']] = word_idx
                    word_idx += 1
            # ro标注转为grid label
            if 'ro_linkings' in json_obj:
                grid_labels = [
                    [0 for j in range(self.max_num_units)]
                    for i in range(self.max_num_units)]
                for segment in json_obj['document']:
                    for wi, wj in zip(segment['words'], segment['words'][1:]):
                        grid_labels[w_id_to_w_idx[wi['id']]][w_id_to_w_idx[wj['id']]] = 1
                for i, j in json_obj['ro_linkings']:
                    wi = json_obj['document'][s_id_to_s_idx[i]]['words'][-1]
                    wj = json_obj['document'][s_id_to_s_idx[j]]['words'][0]
                    grid_labels[w_id_to_w_idx[wi['id']]][w_id_to_w_idx[wj['id']]] = 1
                grid_labels = torch.tensor(grid_labels, dtype=torch.long)
            else:
                grid_labels = None
            # 构造每个unit的mask
            unit_masks = []
            for i, j in word_spans:
                unit_mask = [0] * self.max_seq_len
                unit_mask[i:j] = [1] * (j - i)
                unit_masks.append(unit_mask)
        else:
            raise ValueError(f'self.unit_type is {self.box_level}; expected "segment" or "box"')
        # 确定该条样本有效unit个数，在num_units之外的标签不参与计算loss/metric
        if self.allow_truncate:
            # 试图找到这条样本的标识信息，并打印出来
            if len(unit_masks) > self.max_num_units:
                if 'uid' in json_obj:
                    print(json_obj['uid'])
                elif 'img' in json_obj:
                    print(json_obj['img'])
                print(f'Length exceeds: layout units {len(unit_masks)} > max_num_units {self.max_num_units}. Do truncate.')
            # 截断过长部分
            unit_masks = unit_masks[:self.max_num_units]
        else: 
            assert len(unit_masks) <= self.max_num_units
        global_pointer_masks = torch.tensor([1] * len(unit_masks) + [0] * (self.max_num_units - len(unit_masks)),
                                           dtype=torch.long)
        num_units = torch.tensor(len(unit_masks), dtype=torch.long)

        # unit_mask需要pad到max_num_units个数
        while len(unit_masks) < self.max_num_units:
            # 用来pad的默认unit_mask，不是全0就行，这里设计成只可见cls token
            unit_mask = [0] * self.max_seq_len
            unit_mask[0] = 1
            unit_masks.append(unit_mask)

        unit_masks = torch.tensor(unit_masks, dtype=torch.long)


        # 处理图像部分
        if 'fname' not in json_obj['img']:
            if 'image_path' in json_obj['img']:
                json_obj['img']['fname'] = json_obj['img']['image_path'] # CORD
            elif 'uid' in json_obj:
                json_obj['img']['fname'] = f"images/{json_obj['uid']}.jpg" # SROIE
        img = pil_loader(os.path.join(self.img_dir, json_obj['img']['fname']))
        for_patches, _ = self.common_transform(img)
        images = self.patch_transform(for_patches)
        visual_attention_mask = torch.ones((self.IMAGE_LEN,), dtype=torch.long)
        attention_mask = torch.cat(
            (text_attention_mask, visual_attention_mask), dim=0)
        # unit_masks也扩充图像部分的长度，方便取实体表示
        unit_masks = torch.cat(
            (unit_masks, torch.zeros((self.max_num_units, self.IMAGE_LEN), dtype=torch.long)),
            dim=1)

        # return
        ret = {
            'images': images,  # (3, 224, 224) == (3, image_input_size, image_input_size)
            'input_ids': input_ids,  # (2048,) == (max_seq_len,)
            'attention_mask': attention_mask,  # (2048+197,) == (max_seq_len+IMAGE_LEN,)
            'bboxes': bboxes,  # (2048, 4) == (max_seq_len, 4)
            'num_units': num_units,  # scalar tensor
            'global_pointer_masks': global_pointer_masks, # (256,) == (max_num_units,)
            'unit_masks': unit_masks,  # (256, 2048) == (max_num_units, max_seq_len)
        }
        if grid_labels is not None:
            ret['grid_labels'] = grid_labels # (256, 256) == (max_num_units, max_num_units)
        return ret

