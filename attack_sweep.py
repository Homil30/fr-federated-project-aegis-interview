# attack_sweep.py
"""
Run a sweep of adversarial PGD attacks with different eps/iters,
save results to CSV, and store adversarial images.

Usage:
    python attack_sweep.py --image photos/personA.jpg --compare photos/personB.jpg --model global_model.pt
"""

import os, csv, argparse, torch
from model import SimpleFaceNet
from adversary import build_art_classifier, pgd_attack
from torch import nn
from attack_eval import load_image_tensor, save_tensor_as_image, safe_load_model

def run_attack(model, classifier, img_path, cmp_path, device, eps, iters, outdir):
    x = load_image_tensor(img_path, device=device, resize=(112,112))
    with torch.no_grad():
        emb_clean = model(x)

    adv = pgd_attack(classifier, x, eps=eps, eps_step=eps/10, max_iter=iters)
    adv = adv.to(device)
    with torch.no_grad():
        emb_adv = model(adv)

    sim_clean = torch.nn.functional.cosine_similarity(emb_clean, emb_clean).item()
    sim_adv = torch.nn.functional.cosine_similarity(emb_clean, emb_adv).item()

    # Save adversarial image
    out_img = os.path.join(outdir, f"adv_eps{eps}_it{iters}.png")
    save_tensor_as_image(adv, out_img)
    
    sim_cmp_clean = sim_cmp_adv = None
    if cmp_path:
        x2 = load_image_tensor(cmp_path, device=device, resize=(112,112))
        with torch.no_grad():
            emb_cmp = model(x2)
            sim_cmp_clean = torch.nn.functional.cosine_similarity(emb_clean, emb_cmp).item()
            sim_cmp_adv = torch.nn.functional.cosine_similarity(emb_adv, emb_cmp).item()

    return sim_clean, sim_adv, sim_cmp_clean, sim_cmp_adv, out_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--compare", default=None, help="Optional second identity image")
    parser.add_argument("--model", default="global_model.pt")
    parser.add_argument("--outdir", default="adv_results", help="Dir to save outputs")
    parser.add_argument("--csv", default="attack_sweep.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = safe_load_model(args.model, embedding_size=128, device=device)
    model.eval()
    loss_fn = nn.MSELoss()
    classifier = build_art_classifier(model, loss_fn, input_shape=(3,112,112), nb_classes=100)

    eps_list = [0.01, 0.03, 0.06, 0.1, 0.15]
    iters_list = [5, 10, 20]

    rows = []
    for eps in eps_list:
        for iters in iters_list:
            sc, sa, cmp_c, cmp_a, out_img = run_attack(
                model, classifier, args.image, args.compare, device, eps, iters, args.outdir
            )
            drop = sc - sa
            row = [eps, iters, sc, sa, drop, cmp_c, cmp_a, out_img]
            rows.append(row)
            print(f"eps={eps}, iters={iters}, clean={sc:.4f}, adv={sa:.4f}, drop={drop:.4f}, saved={out_img}")

    # Save CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eps","iters","sim_clean","sim_adv","drop","cmp_clean","cmp_adv","adv_img"])
        writer.writerows(rows)

    print(f"✅ Results saved to {args.csv}, images in {args.outdir}/")
