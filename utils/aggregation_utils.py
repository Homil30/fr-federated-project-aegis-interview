import torch
from collections import OrderedDict

def parameters_to_state_dict(model_template, parameters):
    """Convert Flower parameters back to a PyTorch state_dict"""
    params_dict = zip(model_template.state_dict().keys(), parameters)
    return OrderedDict({k: torch.tensor(v) for k, v in params_dict})
