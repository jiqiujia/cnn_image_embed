# -*- coding: utf-8 -*-
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        cnt = 0
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
                cnt += 1
            if cnt == len(self.extracted_layers):
                break
        return outputs