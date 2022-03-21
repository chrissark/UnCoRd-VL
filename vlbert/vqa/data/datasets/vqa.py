import os
import json
import _pickle as cPickle
from PIL import Image
import re
import base64
import numpy as np
import csv
import sys
import time
import pprint
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

from pycocotools.coco import COCO
from torchvision import transforms

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


class VQA(Dataset):
    def __init__(self, image_path, root, descriptions, answer_vocab, test_mode=False, transform=False):

        super(VQA, self).__init__()

        self.images = list(sorted(os.listdir(os.path.join(root, image_path))))
        self.image_path = image_path
        self.root = root
        self.descriptions = descriptions
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.answer_vocab = answer_vocab
        self.test_mode = test_mode
        self.transform = transform

    def __getitem__(self, index):

        description = self.descriptions[index]
        question = [key for key in description.keys() if key not in ('image_id', 'obj_id', 'rel_obj_id', 'boxes')][0]
        answer = description[question]
        boxes = torch.as_tensor(description['boxes'], dtype=torch.float32).reshape((len(description['boxes']), -1))
        image_id = description['image_id']
        obj_id = description['obj_id']
        if len(boxes) > 1:
            rel_obj_id = description['rel_obj_id']
        img_path = os.path.join(self.root, self.image_path, self.images[image_id])
        image = Image.open(img_path).convert("RGB")
        w0, h0 = image.size
        im_info = torch.tensor([w0, h0, 1.0, 1.0])

        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)

        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # tokenization
        q_tokens = self.tokenizer.tokenize(question)
        q_ids = self.tokenizer.convert_tokens_to_ids(q_tokens)

        # answers
        soft_target = torch.zeros(len(self.answer_vocab), dtype=torch.float)
        answer_index = self.answer_to_ind(answer)
        if answer != self.answer_vocab.index('<unk>'):
            soft_target[answer_index] = 1

        label = soft_target

        if self.test_mode:
            return image, boxes, im_info, q_ids
        else:
            return image, boxes, im_info, q_ids, label

    def __len__(self):
        return len(self.descriptions)

    def answer_to_ind(self, answer):
        if answer in self.answer_vocab:
            return self.answer_vocab.index(answer)
        else:
            return self.answer_vocab.index('<unk>')

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'question']
        else:
            return ['image', 'boxes', 'im_info', 'question', 'label']

