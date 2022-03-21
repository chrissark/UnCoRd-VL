from vlbert.vqa.function.config import config, update_config
import os
import torch
from vlbert.common.utils.load import smart_load_model_state_dict, smart_partial_load_model_state_dict
from vlbert.vqa.modules import *
from vlbert.vqa.data.transforms.build import build_transforms
from vlbert.external.pytorch_pretrained_bert import BertTokenizer
from vlbert.common.utils.clip_pad import *
import numpy as np


def load_vlbert(ckpt_path, device):
    cfgs = 'vlbert/cfgs/vqa/base_4x16G_fp32.yaml'
    update_config(cfgs)
    os.chdir("vlbert/")

    # get model
    model = eval(config.MODULE)(config)

    if config.NETWORK.PARTIAL_PRETRAIN != "":
        pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=device)['state_dict']
        prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
        if len(prefix_change) > 0:
            pretrain_state_dict_parsed = {}
            for k, v in pretrain_state_dict.items():
                no_match = True
                for pretrain_prefix, new_prefix in prefix_change:
                    if k.startswith(pretrain_prefix):
                        k = new_prefix + k[len(pretrain_prefix):]
                        pretrain_state_dict_parsed[k] = v
                        no_match = False
                        break
                if no_match:
                    pretrain_state_dict_parsed[k] = v
            pretrain_state_dict = pretrain_state_dict_parsed
        smart_partial_load_model_state_dict(model, pretrain_state_dict)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        smart_load_model_state_dict(model, checkpoint['state_dict'])

    #transforms
    transform = build_transforms(config, 'train')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print('Loaded checkpoint.')
    os.chdir("..")

    return model, transform, tokenizer


def get_prediction(image, boxes, question, answer_vocab, tokenizer, model, transform=None):

    w0, h0 = image.size
    im_info = torch.tensor([w0, h0, 1.0, 1.0])
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(len(boxes), -1)
    flipped = False
    if transform is not None:
        image, boxes, _, im_info, flipped = transform(image, boxes, None, im_info, flipped)

    w = im_info[0].item()
    h = im_info[1].item()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

    # tokenization
    q_tokens = tokenizer.tokenize(question)
    q_ids = tokenizer.convert_tokens_to_ids(q_tokens)

    #collate
    max_boxes = len(boxes)
    max_question_length = len(q_ids)

    out = {}
    out['image'] = image
    out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)
    out['question'] = clip_pad_1d(q_ids, max_question_length, pad=0)
    out['im_info'] = torch.as_tensor(im_info)

    image = torch.unsqueeze(out['image'], dim=0)
    boxes = torch.unsqueeze(out['boxes'], dim=0)
    im_info = torch.unsqueeze(im_info, dim=0)
    q_ids = torch.unsqueeze(out['question'], dim=0)
    model.eval()
    output = model(image=image, boxes=boxes, im_info=im_info, question=q_ids)
    answer_id = output['label_logits'].argmax(dim=1).detach().cpu().tolist()
    answer = answer_vocab[answer_id[0]]

    return answer