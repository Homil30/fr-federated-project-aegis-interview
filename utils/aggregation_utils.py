import torch

def parameters_to_state_dict(parameters, model_state_dict):
    """
    Convert Flower parameters (list of ndarrays) to a PyTorch state_dict.
    
    Args:
        parameters: List of NumPy arrays from Flower server
        model_state_dict: Reference state_dict from the model (to match keys)
    
    Returns:
        dict: PyTorch state_dict ready to load into model
    """
    # Zip parameters with the model's keys
    params_dict = zip(model_state_dict.keys(), parameters)
    
    # Convert to torch tensors with same shape/dtype
    return {k: torch.tensor(v, dtype=model_state_dict[k].dtype) for k, v in params_dict}
