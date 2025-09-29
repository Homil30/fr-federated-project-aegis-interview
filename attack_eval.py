# attack_eval.py
import argparse
import torch
import os
from model import SimpleFaceNet
from adversary import build_art_classifier, pgd_attack
from torch import nn
from PIL import Image
import torchvision.transforms as T

# --- Helpers ---
def load_image_tensor(path, device="cpu", resize=(112,112)):
    """
    Load image and return tensor shape (1,3,H,W) in float.
    Returns values in [0,1].
    """
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(resize),
        T.ToTensor(),  # floats in [0,1]
    ])
    t = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return t

def save_tensor_as_image(tensor, out_path, try_unnormalize=True):
    """
    Save a tensor (1,3,H,W) to PNG. Heuristic unnormalization:
    - If tensor.min() < -0.5, assume ImageNet normalization was applied and invert.
    - Otherwise clamp to [0,1] and save.
    """
    t = tensor.detach().cpu().squeeze(0).clamp(-5, 5)  # (3,H,W)
    vmin, vmax = float(t.min()), float(t.max())
    if try_unnormalize and vmin < -0.5 and vmax <= 1.5:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        t = (t * std) + mean
    t = t.clamp(0,1)
    to_pil = T.ToPILImage()
    img = to_pil(t)
    img.save(out_path)

def safe_load_model(model_path, embedding_size=128, device="cpu"):
    model = SimpleFaceNet(embedding_size, pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # Support several common save formats
        if isinstance(state, (list, tuple)):
            keys = list(model.state_dict().keys())
            params = {k: torch.tensor(v) for k, v in zip(keys, state)}
            model.load_state_dict(params)
        elif isinstance(state, dict):
            # look for nested keys
            for k in ["state_dict", "model_state_dict"]:
                if k in state:
                    model.load_state_dict(state[k])
                    break
            else:
                # maybe already a state dict but shapes mismatched -> raise
                model.load_state_dict(state)
        else:
            raise
    model.eval()
    return model

# --- Main (standalone usage) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image (to attack)")
    parser.add_argument("--model", default="global_model.pt", help="Path to saved model")
    parser.add_argument("--out_image", default="adv_example.png", help="Output adversarial image path")
    parser.add_argument("--eps", type=float, default=0.03, help="PGD epsilon")
    parser.add_argument("--eps_step", type=float, default=0.007, help="PGD step per iter")
    parser.add_argument("--iters", type=int, default=10, help="PGD iterations")
    parser.add_argument("--compare", default=None, help="Optional: path to second image to compare embeddings")
    parser.add_argument("--device", default="cpu", help="device (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[attack_eval] Device: {device}")

    model = safe_load_model(args.model, embedding_size=128, device=device)
    model.eval()

    loss_fn = nn.MSELoss()
    classifier = build_art_classifier(model, loss_fn, input_shape=(3,112,112), nb_classes=100)

    x = load_image_tensor(args.image, device=device, resize=(112,112))
    # If your training used ImageNet normalization, uncomment the following:
    # imagenet_norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    # x_model = imagenet_norm(x.squeeze(0)).unsqueeze(0).to(device)
    x_model = x  # change to x_model = imagenet_norm(...) if needed

    with torch.no_grad():
        emb_clean = model(x_model)

    adv = pgd_attack(classifier, x_model, eps=args.eps, eps_step=args.eps_step, max_iter=args.iters)
    adv = adv.to(device)

    save_tensor_as_image(adv, args.out_image, try_unnormalize=True)
    print(f"[attack_eval] Saved adversarial image to: {args.out_image}")

    with torch.no_grad():
        emb_adv = model(adv)

    sim = torch.nn.functional.cosine_similarity(emb_clean, emb_adv).item()
    print(f"[attack_eval] Cosine similarity (clean vs adv): {sim:.6f}")

    if args.compare:
        x2 = load_image_tensor(args.compare, device=device, resize=(112,112))
        x2_model = x2
        with torch.no_grad():
            emb_cmp = model(x2_model)
            sim_clean_cmp = torch.nn.functional.cosine_similarity(emb_clean, emb_cmp).item()
            sim_adv_cmp = torch.nn.functional.cosine_similarity(emb_adv, emb_cmp).item()
        print(f"[attack_eval] Clean vs compare: {sim_clean_cmp:.6f}")
        print(f"[attack_eval] Adv   vs compare: {sim_adv_cmp:.6f}")
        print(f"[attack_eval] Change vs compare: {sim_clean_cmp - sim_adv_cmp:.6f}")

if __name__ == "__main__":
    main()
