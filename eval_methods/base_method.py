import torch.nn as nn


class BaseEvalMethod(nn.Module):
    def __init__(self):
        super(BaseEvalMethod, self).__init__()
        pass
    def vis_processors(self):
        pass
    def generate(self):
        pass