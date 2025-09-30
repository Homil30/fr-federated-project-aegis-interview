"""
Download and prepare FairFace dataset
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def setup_fairface_dataset():
    """Download and setup FairFace dataset"""
    
    project_root = Path.cwd()
    data_dir = project_root / "data" / "fairface"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FAIRFACE DATASET SETUP")
    print("="*60)
    
    # FairFace dataset URLs (from official repo)
    urls = {
        "train": "https://github.com/dchen236/FairFace/raw/master/fairface_label_train.csv",
        "val": "https://github.com/dchen236/FairFace/raw/master/fairface_label_val.csv",
    }
    
    print("\n1. Downloading FairFace labels...")
    
    for split, url in urls.items():
        dest = data_dir / f"fairface_label_{split}.csv"
        if dest.exists():
            print(f"  ✓ {split} labels already exist")
        else:
            print(f"  Downloading {split} labels...")
            try:
                download_file(url, dest)
                print(f"  ✓ Downloaded {split} labels")
            except Exception as e:
                print(f"  ✗ Error downloading {split}: {e}")
    
    print("\n" + "="*60)
    print("IMPORTANT: Image Download Instructions")
    print("="*60)
    print("\nThe FairFace images are large (~10GB total).")
    print("You have two options:\n")
    
    print("Option 1: Download from Official Source")
    print("-" * 40)
    print("1. Visit: https://github.com/dchen236/FairFace")
    print("2. Download the image archives:")
    print("   - train.tar.gz (~7GB)")
    print("   - val.tar.gz (~3GB)")
    print(f"3. Extract to: {data_dir / 'images'}")
    
    print("\nOption 2: Use Subset for Quick Testing")
    print("-" * 40)
    print("For quick testing, you can use a smaller subset:")
    print("1. Download first 1000 images from each split")
    print("2. This will be sufficient for project demonstration")
    
    print("\n" + "="*60)
    print(f"Setup directory: {data_dir}")
    print("="*60)
    
    return data_dir

if __name__ == "__main__":
    data_dir = setup_fairface_dataset()
    
    print("\nNext steps:")
    print("1. Download images (see instructions above)")
    print(f"2. Place images in: {data_dir / 'images'}")
    print("3. Run: python prepare_test_data.py")