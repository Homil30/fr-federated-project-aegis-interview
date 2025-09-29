# utils/save_load.py
import torch
import os
from datetime import datetime

def save_global_model(model, out_path):
    """
    Save PyTorch model state_dict safely with metadata.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    data = {
        'state_dict': model.state_dict(),
        'timestamp': datetime.utcnow().isoformat()
    }
    torch.save(data, out_path)
    print(f"[save_global_model] Saved global model: {out_path}")

def load_global_model(model, path, device='cpu'):
    """
    Load a checkpoint into model. Accepts either {'state_dict': ...} or state_dict directly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_location = torch.device(device)
    chk = torch.load(path, map_location=map_location)
    if isinstance(chk, dict) and 'state_dict' in chk:
        state = chk['state_dict']
    elif isinstance(chk, dict):
        state = chk
    else:
        # If someone saved a full model object, try to adapt
        try:
            model.load_state_dict(chk.state_dict())
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError("Unsupported checkpoint format") from e
    model.load_state_dict(state)
    model.eval()
    print(f"[load_global_model] Loaded checkpoint from {path}")
    return model
