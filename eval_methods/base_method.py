import torch.nn as nn


class BaseEvalMethod(nn.Module):
    def __init__(self):
        super(BaseEvalMethod, self).__init__()
        # Here should be the initialization of the method and its components.
        pass
    def vis_processors(self):
        # Here should be the image data transformation.
        pass
    def generate(self):
        # Here should be the generation function (function to ask a model an specific question regarding an image).
        pass