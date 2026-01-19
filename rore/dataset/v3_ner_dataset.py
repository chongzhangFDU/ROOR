import itertools
import json
import os
from PIL import Image
import numpy as np
import torch
import tqdm
import pickle
from transformers import BartTokenizer, BartConfig
from model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from model.layoutlm_v3.tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from utils.image_utils import RandomResizedCropAndInterpolationWithTwoPic
from utils.utils import transitive_closure_dfs
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from utils.tensor_utils import strings_to_tensor
from multiprocessing.pool import ThreadPool
from threading import Lock


class LayoutLMv3Dataset(Dataset):
    def __init__(self,
                 json_objs, image_dir,
                 layoutlmv3_tokenizer,
                 layoutlmv3_config,
                 encoder_max_length=None,
                 use_image=True,
                 box_level='segment',
                 use_aux_ro=False,
                 transitive_expand=False,
                 is_train_val=True,
                 ner_labels=None,
                 do_multi_row_eval=False,
                 ):

        # 相关配置
        self.image_dir = image_dir
        self.layoutlmv3_tokenizer = layoutlmv3_tokenizer
        self.layoutlmv3_config = layoutlmv3_config
        self.use_image = use_image
        self.box_level = box_level
        self.use_aux_ro = use_aux_ro
        self.transitive_expand = transitive_expand
        self.is_train_val = is_train_val
        self.do_multi_row_eval = do_multi_row_eval
        if encoder_max_length is None:
            self.encoder_max_length = self.layoutlmv3_config.max_position_embeddings - 2
        else:
            self.encoder_max_length = encoder_max_length

        # 图像处理
        if self.use_image:
            # 图像大小，patch大小，patch token长度
            self.image_patch_size = 16
            self.IMAGE_LEN = int(self.layoutlmv3_config.input_size / self.image_patch_size) ** 2 + 1
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
                    size=self.layoutlmv3_config.input_size, interpolation='bicubic'),
            ])
            self.patch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])

        # 序列标注NER
        self.ner_labels = ['O']
        if ner_labels:
            for l in ner_labels:
                self.ner_labels.append(f'B-{l}')
                self.ner_labels.append(f'I-{l}')

        # 预处理每条输入数据
        self.dataset = [self.process(o) for o in tqdm.tqdm(json_objs)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def process_layout_for_layoutlmv3_encoder(self, json_obj):
        """ 返回layoutlmv3的输入格式
            {
                'input_ids': torch.zeros(1, 512).long(),
                'attention_mask': torch.zeros(1, 709).long(), # 512 + 197
                'bbox': torch.zeros(1, 512, 4).long(),
                'images': torch.randn(1, 3, 224, 224),
            }
        """
        # 处理文和框
        input_ids, attention_mask, bbox = [self.layoutlmv3_tokenizer.cls_token_id], [1], [[0, 0, 0, 0]]
        word_id_to_token_span = dict()
        for segment in json_obj['document']:

            if self.box_level == 'real_scan':
                # 模拟随机打断情况，计算这种情况下每个word的segment box
                spans = []
                curr_head = None
                last_center = None
                for i, word in enumerate(segment['words']):
                    box = word['box']
                    if curr_head is None: 
                        curr_head = i 
                    else:
                        if (box[0]+box[2])/2 < last_center[0]:
                            spans.append((curr_head, i))
                            curr_head = i 
                        elif random.random() > 0.9:
                            spans.append((curr_head, i))
                            curr_head = i 
                    last_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                else:
                    spans.append((curr_head, len(segment['words'])))
                # for span, next_span in zip(spans, spans[1:]): assert span[1] == next_span[0]
                # for span in spans: assert span[0] < span[1]
                # assert spans[0][0] == 0
                # assert spans[-1][-1] == len(segment['words'])
                ret_boxes = [[0,0,0,0] for word in segment['words']]
                for span in spans:
                    new_seg_box = [
                        min([segment['words'][i]['box'][0] for i in range(*span)]),
                        min([segment['words'][i]['box'][1] for i in range(*span)]),
                        max([segment['words'][i]['box'][2] for i in range(*span)]),
                        max([segment['words'][i]['box'][3] for i in range(*span)])]
                    for i in range(4):
                        new_seg_box[i] += round(np.random.normal(loc=0, scale=random.randint(5, 10)))
                    for i in range(*span):
                        ret_boxes[i] = new_seg_box 

            for word in segment['words']:
                tokens = self.layoutlmv3_tokenizer(word['text'], add_special_tokens=False).input_ids
                if self.box_level == 'segment':
                    box = segment['box']
                elif self.box_level == 'real_scan':
                    box = ret_boxes[i]
                else:
                    box = word['box']
                max_2d = self.layoutlmv3_config.max_2d_position_embeddings - 2
                box = [
                    round(box[0] * max_2d / json_obj['img']['width']),
                    round(box[1] * max_2d / json_obj['img']['height']),
                    round(box[2] * max_2d / json_obj['img']['width']),
                    round(box[3] * max_2d / json_obj['img']['height']),
                ]
                box = [max(min(b, max_2d), 0) for b in box]
                if len(input_ids) + len(tokens) > self.encoder_max_length:
                    break
                word_id_to_token_span[word['id']] = (len(input_ids), len(input_ids) + len(tokens))
                input_ids += tokens
                attention_mask += [1] * len(tokens)
                bbox += [box] * len(tokens)
        # pad到指定长度，转为tensor
        ori_length = len(input_ids) # 文本部分真实长度，token classification使用
        pad_length = self.encoder_max_length - ori_length
        assert pad_length >= 0
        input_ids.extend([self.layoutlmv3_tokenizer.pad_token_id] * pad_length)
        attention_mask.extend([0] * pad_length)
        bbox.extend([[0, 0, 0, 0]] * pad_length)

        return_dict = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'bbox': torch.tensor(bbox).long(),
            'ori_length': torch.tensor(ori_length).long(),
        }

        # 处理图
        if self.use_image:
            img = pil_loader(os.path.join(self.image_dir, json_obj['img']['fname']))
            for_patches, _ = self.common_transform(img)
            images = self.patch_transform(for_patches)
            return_dict['attention_mask'] = torch.cat(
                (return_dict['attention_mask'], torch.ones((self.IMAGE_LEN,))), dim=0)
            return_dict['images'] = images

        # 处理追加的阅读顺序信号
        if self.use_aux_ro and 'ro_linkings' in json_obj:
            ro_linkings = json_obj['ro_linkings']
            return_dict["ro_attn"] = np.zeros(
                (self.encoder_max_length + self.IMAGE_LEN, self.encoder_max_length + self.IMAGE_LEN),
                dtype=np.float32)
            if self.transitive_expand: ro_linkings = transitive_closure_dfs(ro_linkings)
            for i, j in ro_linkings:
                if i in word_id_to_token_span and j in word_id_to_token_span:
                    i1, i2 = word_id_to_token_span[i]
                    j1, j2 = word_id_to_token_span[j]
                    return_dict["ro_attn"][i1:i2][j1:j2] = 1.0
            return_dict["ro_attn"] = torch.from_numpy(return_dict["ro_attn"])

        # 最终返回
        return return_dict, word_id_to_token_span

    def process_ner_bio(self, json_obj, word_id_to_token_span):
        labels = torch.zeros((self.encoder_max_length,)).long()
        labels[0] = -100
        if 'label_entities' in json_obj:
            for e in json_obj['label_entities']:
                entity_span = (self.encoder_max_length+1, -1)
                for word_idx in e['word_idx']:
                    entity_span = (
                        min(word_id_to_token_span[word_idx][0], entity_span[0]),
                        max(word_id_to_token_span[word_idx][1], entity_span[1]))
                labels[entity_span[0]:entity_span[1]] = self.ner_labels.index(f"I-{e['label']}")
                labels[entity_span[0]] = self.ner_labels.index(f"B-{e['label']}")
                if self.do_multi_row_eval:
                    if 'is_multi_row' in e and e['is_multi_row']:
                        labels_multi_row[entity_span[0]:entity_span[1]] = 1
        if self.use_image:
            labels = torch.cat((labels, torch.full((self.IMAGE_LEN,), -100)), dim=0)
        return labels

    def process(self, json_obj):
        inputs, word_id_to_token_span = self.process_layout_for_layoutlmv3_encoder(json_obj)
        if self.is_train_val:
            inputs['labels'] = self.process_ner_bio(json_obj, word_id_to_token_span)
        return inputs

