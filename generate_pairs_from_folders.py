import os, csv, random, argparse
from itertools import combinations

def collect_images(root):
    # root/<identity>/*.jpg
    identity_to_imgs = {}
    for idname in os.listdir(root):
        d = os.path.join(root, idname)
        if not os.path.isdir(d):
            continue
        imgs = [os.path.join(root, idname, f).replace("\\","/") 
                for f in os.listdir(d) if f.lower().endswith((".jpg",".jpeg",".png"))]
        if len(imgs) > 1:  # only keep if we have ≥2 for positives
            identity_to_imgs[idname] = imgs
    return identity_to_imgs

def make_pairs(identity_to_imgs, num_pos_per_id=10, num_neg_pairs=1000):
    rows = []
    # positive pairs
    for idname, imgs in identity_to_imgs.items():
        combos = list(combinations(imgs, 2))
        random.shuffle(combos)
        for a, b in combos[:num_pos_per_id]:
            rows.append((a, b, 1, idname))

    # negative pairs
    ids = list(identity_to_imgs.keys())
    if len(ids) < 2:
        raise ValueError("❌ Not enough identities for negatives!")

    for _ in range(num_neg_pairs):
        a_id, b_id = random.sample(ids, 2)
        a = random.choice(identity_to_imgs[a_id])
        b = random.choice(identity_to_imgs[b_id])
        rows.append((a, b, 0, a_id))

    random.shuffle(rows)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos_dir", default="photos")
    parser.add_argument("--out", default="fairface_eval.csv")
    parser.add_argument("--pos_per_id", type=int, default=10)
    parser.add_argument("--neg_pairs", type=int, default=1000)
    args = parser.parse_args()

    identity_to_imgs = collect_images(args.photos_dir)
    if not identity_to_imgs:
        print("❌ No valid identities found under", args.photos_dir)
        return

    rows = make_pairs(identity_to_imgs, args.pos_per_id, args.neg_pairs)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img1","img2","label","group"])
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} pairs to {args.out}")

if __name__ == "__main__":
    main()
