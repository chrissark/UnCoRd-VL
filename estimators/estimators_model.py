import os
import warnings
from os.path import isfile, join
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, models, transforms
from tqdm.notebook import trange, tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import ImageDraw


class Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Size estimator

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP_rel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def init_estimator(model, state_dict):

    model.load_state_dict(torch.load(state_dict, map_location="cpu"))

    return model


def get_prediction(model, prop, img=None, box=None):
    COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
    MATERIALS = ['metal', 'rubber']
    SHAPES = ['cube', 'cylinder', 'sphere']
    SIZES = ['small', 'large']
    RELATIONSHIPS = ['left', 'right', 'front', 'behind']
    answers = ['yes', 'no']

    if img is not None:
        img = img.resize((32, 32))

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform(img)

        with torch.no_grad():
            output = model(img.unsqueeze(0))
    elif box is not None:
        with torch.no_grad():
            output = model(torch.as_tensor(box, dtype=torch.float32).unsqueeze(0))

    _, predictions = torch.max(output, 1)

    index = predictions.tolist()[0]
    if prop == 'color':
        pred = COLORS[index]
    elif prop == 'material':
        pred = MATERIALS[index]
    elif prop == 'shape':
        pred = SHAPES[index]
    elif prop == 'size':
        pred = SIZES[index]
    elif prop in RELATIONSHIPS:
        pred = answers[index]

    #print(f'{prop} pred: {pred}')

    return pred

def get_space_relation(cur_obj, obj, model):

    boxes = [cur_obj['box'], obj['box']]
    with torch.no_grad():
        output = model(torch.as_tensor(boxes, dtype=torch.float32).unsqueeze(0))

    _, predictions = torch.max(output, 1)

    return predictions.tolist()[0]




