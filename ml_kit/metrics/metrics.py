''' Copyright (C) Sean Billings - All Rights Reserved
 	Unauthorized copying of this file, via any medium is strictly prohibited
 	Proprietary and confidential
 	Written by Sean Billings <s.a.f.billings@gmail.com>, August 2019
'''
import torch
from copy import deepcopy

class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels.squeeze())
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels.squeeze())
        return torch.mean((x - y) ** 2)

    def accuracy(self, predictions, labels):
        num_correct = (predictions == labels.squeeze()).sum().float().item()
        return num_correct / len(labels)