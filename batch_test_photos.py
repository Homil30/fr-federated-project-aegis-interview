# batch_test_photos.py 
import os
import requests
import pandas as pd
import itertools
import random
import seaborn as sns
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:9000/compare"  # FastAPI server
THRESHOLD = 0.8  # similarity threshold for same/different classification

# 1. Path to your photos folder
photos_path = r"C:\Users\Asus\Desktop\photos"

# 2. Collect all image file paths
all_images = [os.path.join(photos_path, f) for f in os.listdir(photos_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Found {len(all_images)} images.")

# 3. Take a small random sample for testing (e.g., 20 images)
sample_images = random.sample(all_images, 20)

# 4. Build pairs
pairs = list(itertools.combinations(range(len(sample_images)), 2))

results = []

# 5. Loop over pairs and call API
for i, j in pairs:
    img1_path = sample_images[i]
    img2_path = sample_images[j]

    files = {
        "a": open(img1_path, "rb"),
        "b": open(img2_path, "rb"),
    }
    try:
        resp = requests.post(API_URL, files=files)
        score = resp.json()["score"]
    except Exception as e:
        print("Error with", img1_path, img2_path, e)
        score = None

    decision = None
    if score is not None:
        decision = "same" if score >= THRESHOLD else "different"

    results.append({
        "img1": os.path.basename(img1_path),
        "img2": os.path.basename(img2_path),
        "score": score,
        "decision": decision
    })

# 6. Save results to CSV
df = pd.DataFrame(results)
df.to_csv("photos_compare_results.csv", index=False)
print("Saved results to photos_compare_results.csv")

# 7. Print Top-10 analysis
print("\n=== Analysis ===")
print("Top-10 Most Similar Pairs:")
print(df.sort_values("score", ascending=False).head(10))

print("\nTop-10 Least Similar Pairs:")
print(df.sort_values("score", ascending=True).head(10))

# 8. Optional — similarity heatmap
pivot = df.pivot(index="img1", columns="img2", values="score")
plt.figure(figsize=(12, 10))
sns.heatmap(pivot, cmap="viridis", annot=False)
plt.title("Photos Similarity Heatmap (sampled pairs)")
plt.tight_layout()
plt.savefig("photos_similarity_heatmap.png")
print("Saved photos_similarity_heatmap.png")
