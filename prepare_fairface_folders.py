# prepare_fairface_folders.py (debug version)
import os
import pandas as pd
import shutil

meta_csv = r"C:/Users/Asus/Desktop/XAI_Facial_Recognition/data/raw/FairFace/fairface-label-train.csv"
images_root = r"C:/Users/Asus/Desktop/XAI_Facial_Recognition/data/raw/FairFace/train"
output_root = r"C:/Users/Asus/Desktop/XAI_Facial_Recognition/data/raw/FairFace/train_prepared"

os.makedirs(output_root, exist_ok=True)

df = pd.read_csv(meta_csv)
counts = {}

print(f"📂 images_root = {images_root}")
print(f"📝 CSV rows = {len(df)}")
print("👀 First 5 entries in CSV:")
print(df.head())

for _, row in df.iterrows():
    file = row["file"]          # e.g. "train/1.jpg"
    gender = row["gender"].capitalize()

    # ✅ use only basename
    src = os.path.join(images_root, os.path.basename(file))
    dst_dir = os.path.join(output_root, gender)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(file))

    if os.path.exists(src):
        shutil.copy(src, dst)
        counts[gender] = counts.get(gender, 0) + 1
    else:
        # 🚨 Show missing files
        print(f"❌ Missing: {src}")

print("✅ Dataset prepared under:", output_root)
for gender, c in counts.items():
    print(f"  {gender}: {c} images copied")
