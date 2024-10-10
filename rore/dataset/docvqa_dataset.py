import itertools
import json
import os
from PIL import Image
import cv2
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


class DocvqaDataset(Dataset):
    def __init__(self,
                 json_objs, dataset_dir,
                 layoutlmv3_tokenizer, bart_tokenizer,
                 layoutlmv3_config, bart_config,
                 encoder_max_length=None,
                 decoder_max_length=None,
                 box_level='segment',
                 use_aux_ro=False,
                 transitive_expand=False,
                 is_train_val=True):

        # 相关配置
        self.dataset_dir = dataset_dir
        self.layoutlmv3_tokenizer = layoutlmv3_tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.layoutlmv3_config = layoutlmv3_config
        self.bart_config = bart_config
        self.box_level = box_level
        self.use_aux_ro = use_aux_ro
        self.transitive_expand = transitive_expand
        self.is_train_val = is_train_val
        if encoder_max_length is None:
            self.encoder_max_length = self.layoutlmv3_config.max_position_embeddings - 2
        else:
            self.encoder_max_length = encoder_max_length
        if decoder_max_length is None:
            self.decoder_max_length = self.bart_config.max_position_embeddings
        else:
            self.decoder_max_length = decoder_max_length

        # 图像处理
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

        # 处理每条输入数据
        self.dataset = []
        self.raw_sample = dict()
        self.uid_to_sample = dict()

        for o in json_objs:
            self.dataset.append(o['id'])
            self.raw_sample[o['id']] = o

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            if not self.dataset[idx] in self.uid_to_sample:
                self.uid_to_sample[self.dataset[idx]] = self.process(self.raw_sample[self.dataset[idx]])
            return self.uid_to_sample[self.dataset[idx]]
        except:
            return self.__getitem__((idx+1) % len(self.dataset))

    def process_layout_for_layoutlmv3_encoder(self, image_path, json_path):
        """ 返回layoutlmv3的输入格式
            {
                'input_ids': torch.zeros(1, 512).long(),
                'attention_mask': torch.zeros(1, 709).long(), # 512 + 197
                'bbox': torch.zeros(1, 512, 4).long(),
                'images': torch.randn(1, 3, 224, 224),
            }
        """
        # 处理文和框
        with open(os.path.join(self.dataset_dir, json_path)) as f:
            json_obj = json.load(f)
        input_ids, attention_mask, bbox = [], [], []
        seg_id_to_token_span = dict()
        for segment in json_obj['document']:
            for word in segment['words']:
                tokens = self.layoutlmv3_tokenizer(word['text'], add_special_tokens=False).input_ids
                if self.box_level == 'segment':
                    box = segment['box']
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
                seg_id_to_token_span[segment['id']] = (len(input_ids), len(input_ids) + len(tokens))
                input_ids += tokens
                attention_mask += [1] * len(tokens)
                bbox += [box] * len(tokens)
        # pad到指定长度，转为tensor
        pad_length = self.encoder_max_length - len(input_ids)
        assert pad_length >= 0
        input_ids.extend([self.layoutlmv3_tokenizer.pad_token_id] * pad_length)
        attention_mask.extend([0] * pad_length)
        bbox.extend([[0, 0, 0, 0]] * pad_length)

        # 处理图
        img = pil_loader(os.path.join(self.dataset_dir, image_path))
        for_patches, _ = self.common_transform(img)
        images = self.patch_transform(for_patches)
        attention_mask.extend([1] * self.IMAGE_LEN)

        return_dict = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'bbox': torch.tensor(bbox).long(),
            'images': images,
        }

        # 处理追加的阅读顺序信号
        if self.use_aux_ro and 'ro_linkings' in json_obj:
            ro_linkings = json_obj['ro_linkings']
            return_dict["ro_attn"] = np.zeros(
                (self.encoder_max_length + self.IMAGE_LEN, self.encoder_max_length + self.IMAGE_LEN), 
                dtype=np.float32)
            if self.transitive_expand: ro_linkings = transitive_closure_dfs(ro_linkings)
            for i, j in ro_linkings:
                if i in seg_id_to_token_span and j in seg_id_to_token_span:
                    i1, i2 = seg_id_to_token_span[i]
                    j1, j2 = seg_id_to_token_span[j]
                    return_dict["ro_attn"][i1:i2][j1:j2] = 1.0
            return_dict["ro_attn"] = torch.from_numpy(return_dict["ro_attn"])

        # 最终返回
        return return_dict

    def process_qa_for_bart_decoder(self, question, answer=None):
        if answer is not None:
            # 用于forward
            question_tokens = self.bart_tokenizer(f"Question: {question}", add_special_tokens=False).input_ids
            answer_tokens = self.bart_tokenizer(f"Answer: {answer}", add_special_tokens=False).input_ids
            # decoder_input_ids = question_tokens + [self.bart_config.bos_token_id] + answer_tokens
            # labels = [-100] * len(question_tokens) + answer_tokens + [self.bart_config.eos_token_id]
            decoder_input_ids = [self.bart_config.bos_token_id] + question_tokens + answer_tokens
            labels = question_tokens + answer_tokens + [self.bart_config.eos_token_id]
            decoder_attention_mask = [1] * len(decoder_input_ids)
            assert self.decoder_max_length >= len(decoder_input_ids)
            pad_length = self.decoder_max_length - len(decoder_input_ids)
            decoder_input_ids = [self.bart_config.pad_token_id] * pad_length + decoder_input_ids
            labels = [-100] * pad_length + labels
            decoder_attention_mask = [0] * pad_length + decoder_attention_mask
            decoder_input_ids = torch.tensor(decoder_input_ids).long()
            labels = torch.tensor(labels).long()
            decoder_attention_mask = torch.tensor(decoder_attention_mask).long()
        else:
            # 用于generate
            question_tokens = self.bart_tokenizer(f"Question: {question} Answer: ", add_special_tokens=False).input_ids
            # decoder_input_ids = question_tokens + [self.bart_config.bos_token_id]
            decoder_input_ids = [self.bart_config.bos_token_id] + question_tokens
            decoder_attention_mask = [1] * len(decoder_input_ids)
            assert self.decoder_max_length >= len(decoder_input_ids)
            pad_length = self.decoder_max_length - len(decoder_input_ids)
            decoder_input_ids = [self.bart_config.pad_token_id] * pad_length + decoder_input_ids
            decoder_attention_mask = [0] * pad_length + decoder_attention_mask
            decoder_input_ids = torch.tensor(decoder_input_ids).long()
            labels = None
            decoder_attention_mask = torch.tensor(decoder_attention_mask).long()
        return decoder_input_ids, decoder_attention_mask, labels

    def process(self, json_obj):
        sample = self.process_layout_for_layoutlmv3_encoder(json_obj['image'], json_obj['json'])
        if 'answers' in json_obj:
            # 使用pickle序列化把answers编码成tensor, 用于validation
            tensor, lengths, num_strings = strings_to_tensor(json_obj['answers'])
            sample['answers_raw'] = tensor
            sample['answers_lengths'] = lengths
            sample['answers_num_strings'] = num_strings
        if self.is_train_val and 'answers' in json_obj:
            # 用于forward
            decoder_input_ids, decoder_attention_mask, labels = self.process_qa_for_bart_decoder(json_obj['question'], json_obj['answers'][0])
            sample['decoder_input_ids'] = decoder_input_ids
            sample['decoder_attention_mask'] = decoder_attention_mask
            sample['labels'] = labels
        else:
            # 用于generate，暂时不支持batch decode
            decoder_input_ids, decoder_attention_mask, _ = self.process_qa_for_bart_decoder(json_obj['question'], None)
            sample['decoder_input_ids'] = decoder_input_ids
            sample['decoder_attention_mask'] = decoder_attention_mask
        return sample
