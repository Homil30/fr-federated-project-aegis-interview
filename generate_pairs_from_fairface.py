# generate_pairs_from_fairface.py
import os, csv, random, argparse
import pandas as pd
from collections import defaultdict

def load_metadata(meta_csv, images_root="", group_by="gender_race"):
    df = pd.read_csv(meta_csv)
    rows = []
    for _, r in df.iterrows():
        file = r["file"]

        gender = r.get("gender", "unknown")
        race = r.get("race", "unknown")
        age = r.get("age", "unknown")

        # grouping
        if group_by == "gender":
            group = gender
        elif group_by == "race":
            group = race
        elif group_by == "age":
            group = age
        else:
            group = f"{gender}_{race}"

        path = os.path.join(images_root, file).replace("\\", "/")

        # ✅ use filename prefix as identity (e.g., 0_xxx.jpg → id=0)
        identity = os.path.basename(file).split("_")[0]

        rows.append((path, identity, group))
    return rows


def make_pairs(meta_rows, pos_per_id=5, neg_pairs=100):
    id_to_images = defaultdict(list)
    img_to_group = {}
    for path, identity, group in meta_rows:
        id_to_images[identity].append(path)
        img_to_group[path] = group

    rows = []

    # ✅ synthetic positive pairs (self-pairs)
    for identity, imgs in id_to_images.items():
        for img in imgs[:pos_per_id]:  # add up to pos_per_id self-pairs
            rows.append((img, img, 1, img_to_group[img]))

    # ✅ real negative pairs (different identities)
    ids = list(id_to_images.keys())
    if len(ids) < 2:
        print("❌ Not enough identities for negative pairs")
        return rows

    max_neg = min(neg_pairs, len(ids) * 10)
    for _ in range(max_neg):
        a_id, b_id = random.sample(ids, 2)
        a = random.choice(id_to_images[a_id])
        b = random.choice(id_to_images[b_id])
        rows.append((a, b, 0, img_to_group[a]))

    random.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="FairFace metadata CSV")
    parser.add_argument("--images_root", required=True, help="path to FairFace images")
    parser.add_argument("--out", default="fairface_eval.csv")
    parser.add_argument("--pos_per_id", type=int, default=5)
    parser.add_argument("--neg_pairs", type=int, default=100)
    parser.add_argument("--group_by", choices=["gender", "race", "age", "gender_race"], default="gender_race")
    args = parser.parse_args()

    rows = load_metadata(args.meta, args.images_root, args.group_by)
    pairs = make_pairs(rows, args.pos_per_id, args.neg_pairs)

    with open(args.out, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label", "group"])
        writer.writerows(pairs)

    print(f"✅ Saved {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
