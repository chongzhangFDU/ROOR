import itertools
import json
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
from utils.utils import transitive_closure_dfs


def point2_to_point4(box):
    if isinstance(box[0], list): return box
    return [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]


def point4_to_point2(box):
    if not isinstance(box[0], list): return box
    xs, ys = list(zip(*box))
    return [min(xs), min(ys), max(xs), max(ys)]


class DocumentDataset(Dataset):
    def __init__(self,
                 dataset, img_dir, tokenizer,
                 max_block_num=256, max_seq_length=1024, img_h=768, img_w=768,
                 use_segment=True,
                 use_aux_ro=False, transitive_expand=False,
                 class_names=None):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.backbone_type = ''  # 'layoutlm'
        self.max_seq_length = max_seq_length
        self.max_block_num = max_block_num
        self.img_h = img_h
        self.img_w = img_w
        self.use_segment = use_segment
        self.use_aux_ro = use_aux_ro
        self.transitive_expand = transitive_expand
        if getattr(self.tokenizer, "vocab", None) is not None:
            self.pad_token_id = self.tokenizer.vocab["[PAD]"]
            self.cls_token_id = self.tokenizer.vocab["[CLS]"]
            self.sep_token_id = self.tokenizer.vocab["[SEP]"]
            self.unk_token_id = self.tokenizer.vocab["[UNK]"]
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.cls_token_id = self.tokenizer.cls_token_id
            self.sep_token_id = self.tokenizer.sep_token_id
            self.unk_token_id = self.tokenizer.unk_token_id
        self.class_names = class_names
        self.class_idx_dic = dict(
            [(class_name, idx) for idx, class_name in enumerate(self.class_names)]
        )
        self.bio_class_names = ["O"]
        for class_name in self.class_names:
            if not class_name.startswith('O'):
                self.bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
        self.bio_class_idx_dic = dict(
            [
                (bio_class_name, idx)
                for idx, bio_class_name in enumerate(self.bio_class_names)
            ]
        )
        self.dataset = []
        for json_object in tqdm.tqdm(dataset):
            self.dataset.append(self.process(self.cdip_to_geo(json_object)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def process(self, json_obj):
        return_dict = {}

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        img_path = json_obj["meta"]["image_path"]

        image = cv2.resize(cv2.imread(img_path, 1), (self.img_w, self.img_h))
        image = image.astype("float32").transpose(2, 0, 1)

        # return_dict["image_path"] = img_path
        return_dict["image"] = image
        return_dict["size_raw"] = np.array([width, height])

        return_dict["input_ids"] = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        return_dict["bbox_4p_normalized"] = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        return_dict["attention_mask"] = np.zeros(self.max_seq_length, dtype=int)
        return_dict["first_token_idxes"] = np.zeros(self.max_block_num, dtype=int)
        return_dict["block_mask"] = np.zeros(self.max_block_num, dtype=int)
        return_dict["bbox"] = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        return_dict["line_rank_id"] = np.zeros(self.max_seq_length, dtype="int32")
        return_dict["line_rank_inner_id"] = np.ones(self.max_seq_length, dtype="int32")

        return_dict["are_box_first_tokens"] = np.zeros(self.max_seq_length, dtype=np.bool_)
        return_dict["bio_labels"] = np.zeros(self.max_seq_length, dtype=int)
        return_dict["el_labels_seq"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        return_dict["el_label_seq_mask"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        return_dict["el_labels_blk"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)
        return_dict["el_label_blk_mask"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)

        list_tokens = []
        list_bbs = []  # word boxes
        list_seg_bbs = []  # segment boxes
        box2token_span_map = []

        box_to_token_indices = []
        cum_token_idx = 0

        cls_bbs = [0.0] * 8
        # cls_bbs_blk = [0] * 4

        for word_idx, word in enumerate(json_obj["words"]):
            this_box_token_indices = []

            tokens = word["tokens"]
            bb = word["boundingBox"]
            seg_bb = word["segmentBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            # 原始GeoLayoutLM将长文档截断，不考虑超长部分的实体和这些实体上的关系，造成不公平比较
            # 这里对这一情况进行避免
            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                raise ValueError('Document length exceeds')

            box2token_span_map.append(
                [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
            )  # including st_idx, start from 1
            list_tokens += tokens

            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))
                seg_bb[coord_idx][0] = max(0.0, min(seg_bb[coord_idx][0], width))
                seg_bb[coord_idx][1] = max(0.0, min(seg_bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            seg_bb = list(itertools.chain(*seg_bb))
            bbs = [bb for _ in range(len(tokens))]
            seg_bbs = [seg_bb for _ in range(len(tokens))]

            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx)  # start from 1

            list_bbs.extend(bbs)
            list_seg_bbs.extend(seg_bbs)
            box_to_token_indices.append(this_box_token_indices)

        sep_bbs = [width, height] * 4

        first_token_idx_list = json_obj['blocks']['first_token_idx_list'][:self.max_block_num]
        if first_token_idx_list[-1] > len(list_tokens):
            blk_length = self.max_block_num
            for blk_id, first_token_idx in enumerate(first_token_idx_list):
                if first_token_idx > len(list_tokens):
                    blk_length = blk_id
                    break
            first_token_idx_list = first_token_idx_list[:blk_length]

        first_token_ext = first_token_idx_list + [len(list_tokens) + 1]
        line_id = 1
        for blk_idx in range(len(first_token_ext) - 1):
            token_span = first_token_ext[blk_idx + 1] - first_token_ext[blk_idx]
            # block box
            bb_blk = json_obj['blocks']['boxes'][blk_idx]
            bb_blk[0] = max(0, min(bb_blk[0], width))
            bb_blk[1] = max(0, min(bb_blk[1], height))
            bb_blk[2] = max(0, min(bb_blk[2], width))
            bb_blk[3] = max(0, min(bb_blk[3], height))
            # line_rank_id
            return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx + 1]] = line_id
            line_id += 1
            # line_rank_inner_id
            if token_span > 1:
                return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx + 1]] = [1] + [2] * (
                        token_span - 2) + [3]

        # For [CLS] and [SEP]
        list_tokens = (
                [self.cls_token_id]
                + list_tokens[: self.max_seq_length - 2]
                + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
            list_seg_bbs = [cls_bbs] + [sep_bbs]
        elif len(list_bbs) <= self.max_seq_length - 2:
            list_bbs = [cls_bbs] + list_bbs + [sep_bbs]
            list_seg_bbs = [cls_bbs] + list_seg_bbs + [sep_bbs]
        else:
            raise ValueError('Sequence length exceeds')

        len_list_tokens = len(list_tokens)
        # len_blocks = len(first_token_idx_list)
        return_dict["input_ids"][:len_list_tokens] = list_tokens
        return_dict["attention_mask"][:len_list_tokens] = 1

        return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

        bbox_4p_normalized = return_dict["bbox_4p_normalized"]
        bbox_4p_normalized[:len_list_tokens, :] = list_bbs

        # bounding box normalization -> [0, 1]
        bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
        bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

        if self.backbone_type == "layoutlm":
            bbox_4p_normalized = bbox_4p_normalized[:, [0, 1, 4, 5]]
            bbox_4p_normalized = bbox_4p_normalized * 1000
            bbox_4p_normalized = bbox_4p_normalized.astype(int)

        return_dict["bbox_4p_normalized"] = bbox_4p_normalized

        bbox = return_dict["bbox"]
        bbox[:len_list_tokens, :] = list_seg_bbs

        # bounding box normalization -> [0, 1]
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height
        bbox = bbox[:, [0, 1, 4, 5]]
        bbox = bbox * 1000
        bbox = bbox.astype(int)
        return_dict["bbox"] = bbox

        st_indices = [
            indices[0]
            for indices in box_to_token_indices
            if indices[0] < self.max_seq_length
        ]
        return_dict["are_box_first_tokens"][st_indices] = True

        # Label for tagging
        classes_dic = json_obj["parse"]["class"]
        for class_name in self.class_names:
            # if class_name == "O":
            #     continue
            if class_name not in classes_dic:
                continue

            for word_list in classes_dic[class_name]:
                # At first, connect the class and the first box
                is_first, last_word_idx = True, -1
                for word_idx in word_list:
                    assert word_idx < len(box2token_span_map)
                    box2token_span_start, box2token_span_end = \
                        box2token_span_map[word_idx]
                    for converted_word_idx in range(
                            box2token_span_start, box2token_span_end
                    ):
                        assert converted_word_idx < self.max_seq_length

                        if class_name == 'O':
                            return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
                                "O"
                            ]
                            continue
                        if is_first:
                            return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
                                f"B-{class_name}"
                            ]
                            is_first = False
                        else:
                            return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
                                f"I-{class_name}"
                            ]
        return_dict["bio_labels"][0] = -100
        return_dict["bio_labels"][len_list_tokens:] = -100
        # Label for linking
        relations = json_obj["parse"]["relations"]
        ent_first_token_idx = json_obj["parse"]["ent_first_token_idx"]
        len_ents = len(ent_first_token_idx)
        return_dict["first_token_idxes"][:len(ent_first_token_idx)] = ent_first_token_idx
        return_dict["block_mask"][:len_ents] = 1
        for relation in relations:
            assert relation[0] < len(box2token_span_map)
            assert relation[1] < len(box2token_span_map)
            assert box2token_span_map[relation[0]][0] < self.max_seq_length
            assert box2token_span_map[relation[1]][0] < self.max_seq_length
            # word_from = box2token_span_map[relation[0]][0]
            # word_to = box2token_span_map[relation[1]][0]
            word_from = ent_first_token_idx[relation[0]]
            word_to = ent_first_token_idx[relation[1]]
            return_dict["el_labels_seq"][word_to][word_from] = 1.0
            return_dict["el_labels_blk"][relation[1]][relation[0]] = 1.0
        return_dict["el_label_seq_mask"][1:len_list_tokens, 1:len_list_tokens] = 1.0
        # B_idx_all = np.array(first_token_idx_list)
        # return_dict["el_labels_blk"][:len(first_token_idx_list), :len(first_token_idx_list)] = \
        #     return_dict["el_labels_seq"][B_idx_all[:, np.newaxis], B_idx_all[np.newaxis, :]]
        return_dict["el_label_blk_mask"][:len_ents, :len_ents] = 1.0

        # 构造辅助信号
        if self.use_aux_ro:
            ent_last_token_idx = json_obj["parse"]["ent_last_token_idx"]
            return_dict["ro_attn"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
            ro_linkings = json_obj["parse"]["ro_relations"]
            if self.transitive_expand: ro_linkings = transitive_closure_dfs(ro_linkings)
            for i, j in ro_linkings:
                i1, i2 = ent_first_token_idx[i], ent_last_token_idx[i] + 1
                j1, j2 = ent_first_token_idx[j], ent_last_token_idx[j] + 1
                return_dict["ro_attn"][i1:i2][j1:j2] = 1.0

        # return
        for k in return_dict.keys():
            if isinstance(return_dict[k], np.ndarray):
                return_dict[k] = torch.from_numpy(return_dict[k])

        return return_dict

    def cdip_to_geo(self, json_obj):
        # 构造words和blocks
        word_id_to_word_idx = dict()  # 从词id到词下标
        curr_idx = 0
        word_idx_to_token_idx = dict()  # 从词下标到词token span
        curr_token_idx = 1  # 考虑cls token 所以从1开始计数
        blocks = {'first_token_idx_list': [], 'boxes': []}
        words = []
        if self.use_segment:
            for segment in json_obj['document']:
                blocks['first_token_idx_list'].append(curr_token_idx)
                blocks['boxes'].append(point4_to_point2(segment['box']))
                for word in segment['words']:
                    tokens = self.tokenizer(word['text'], add_special_tokens=False).input_ids
                    words.append({
                        "text": word['text'],
                        "tokens": tokens,
                        "boundingBox": point2_to_point4(word['box']),
                        "segmentBox": point2_to_point4(segment['box'])})
                    word_id_to_word_idx[word['id']] = curr_idx
                    word_idx_to_token_idx[curr_idx] = (curr_token_idx, curr_token_idx + len(tokens))
                    curr_idx += 1
                    curr_token_idx += len(tokens)
        else:
            for segment in json_obj['document']:
                for word in segment['words']:
                    blocks['first_token_idx_list'].append(curr_token_idx)
                    blocks['boxes'].append(point4_to_point2(word['box']))
                    tokens = self.tokenizer(word['text'], add_special_tokens=False).input_ids
                    words.append({
                        "text": word['text'],
                        "tokens": tokens,
                        "boundingBox": point2_to_point4(word['box']),
                        "segmentBox": point2_to_point4(word['box'])})
                    word_id_to_word_idx[word['id']] = curr_idx
                    word_idx_to_token_idx[curr_idx] = (curr_token_idx, curr_token_idx + len(tokens))
                    curr_idx += 1
                    curr_token_idx += len(tokens)
        # 构造parse-class
        # 从实体构造class
        # entities = {"O": [], "HEADER": [], "QUESTION": [], "ANSWER": []}
        entities = {class_name: [] for class_name in self.class_names}
        registered_words = []  # 记录登记为实体的words；没有登记为实体的words要作为O输入模型
        for e in json_obj['label_entities']:
            if isinstance(e['word_idx'][0], list):
                e['word_idx'] = [idx for ls in e['word_idx'] for idx in ls]
            idxs = [word_id_to_word_idx[i] for i in e['word_idx']]
            try:
                entities[e['label']].append(idxs)
            except Exception as e:
                raise ValueError(f"Unknown entity type: {e['label']} in {self.class_names}")
            for w in idxs:
                if not w in registered_words: registered_words.append(w)
        for i in range(curr_idx):
            if not i in registered_words: entities['O'].append([i])
        # 构造parse-relations
        # 列表first_token_idxs 从前往后每个实体第一个词第一个token的idx
        # 列表last_token_idxs 从前往后每个实体最后一个词最后一个token的idx
        # 实体idxpair的列表 relations 表示关系
        # 实体id to 实体第一个词的idx
        ent_id_to_ent_idx = dict()
        ent_first_token_idx = []
        ent_last_token_idx = []
        for i, e in enumerate(json_obj['label_entities']):
            ent_id_to_ent_idx[e['entity_id']] = i
            ent_first_token_idx.append(
                word_idx_to_token_idx[word_id_to_word_idx[e['word_idx'][0]]][0])
            ent_last_token_idx.append(
                word_idx_to_token_idx[word_id_to_word_idx[e['word_idx'][-1]]][-1])
        # 从关系构造relations
        relations = []
        if 'label_linkings' in json_obj:
            for i, j in json_obj['label_linkings']:
                if i in ent_id_to_ent_idx and j in ent_id_to_ent_idx:
                    relations.append([ent_id_to_ent_idx[i], ent_id_to_ent_idx[j]])
        relations.sort()
        # 辅助关系信号
        ro_relations = []
        if 'ro_linkings' in json_obj and self.use_aux_ro:
            for i, j in json_obj['ro_linkings']:
                if i in ent_id_to_ent_idx and j in ent_id_to_ent_idx:
                    ro_relations.append([ent_id_to_ent_idx[i], ent_id_to_ent_idx[j]])
        ro_relations.sort()
        parse = {'class': entities, 'relations': relations, 'ro_relations': ro_relations,
                 'ent_first_token_idx': ent_first_token_idx, 'ent_last_token_idx': ent_last_token_idx}
        # 构造meta，使用上游传入的绝对路径
        if 'fname' not in json_obj['img']:
            if 'image_path' in json_obj['img']:
                json_obj['img']['fname'] = json_obj['img']['image_path']  # 支持CORD
            elif 'uid' in json_obj:
                json_obj['img']['fname'] = f"images/{json_obj['uid']}.jpg"  # 支持SROIE
        meta = {'image_path': os.path.join(self.img_dir, json_obj['img']['fname']),
                'imageSize': {'width': json_obj['img']['width'], 'height': json_obj['img']['height']},
                'voca': 'bert-base-uncased'}
        # return
        return {'blocks': blocks, 'words': words, 'parse': parse, 'meta': meta}

