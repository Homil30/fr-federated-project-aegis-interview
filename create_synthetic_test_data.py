"""
Create synthetic test data for immediate evaluation
This allows you to complete the project without waiting for large downloads
"""
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import random

def create_synthetic_face_image(gender, race, size=(224, 224)):
    """Create a synthetic face-like image with variation"""
    img = np.random.randint(80, 180, (*size, 3), dtype=np.uint8)
    
    # Add some structure to make it more face-like
    center_y, center_x = size[0] // 2, size[1] // 2
    y, x = np.ogrid[:size[0], :size[1]]
    
    # Create face oval shape
    face_mask = ((x - center_x)**2 / (size[1]//3)**2 + 
                 (y - center_y)**2 / (size[0]//2.5)**2) <= 1
    
    # Different color tints based on attributes
    if gender == 'Male':
        img[:, :, 0] = img[:, :, 0] * 0.9  # Less red
    else:
        img[:, :, 0] = img[:, :, 0] * 1.1  # More red
    
    # Apply face mask
    img[~face_mask] = img[~face_mask] * 0.7
    
    return Image.fromarray(img.astype(np.uint8))

def create_test_dataset(num_samples_per_class=100):
    """Create synthetic test dataset"""
    
    project_root = Path.cwd()
    test_dir = project_root / "data" / "test_synthetic"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CREATING SYNTHETIC TEST DATA")
    print("="*60)
    
    demographics = {
        'gender': ['Male', 'Female'],
        'race': ['White', 'Black', 'Asian', 'Indian', 'Latino_Hispanic'],
        'age': ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    }
    
    all_data = []
    image_id = 0
    
    print(f"\nGenerating {num_samples_per_class} samples per demographic group...")
    
    for gender in demographics['gender']:
        for race in demographics['race']:
            # Create folder structure
            class_dir = test_dir / gender / race
            class_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Creating {gender}/{race}...")
            
            for i in range(num_samples_per_class):
                # Generate image
                img = create_synthetic_face_image(gender, race)
                
                # Save image
                img_filename = f"{gender}_{race}_{i:04d}.jpg"
                img_path = class_dir / img_filename
                img.save(img_path)
                
                # Record metadata
                age = random.choice(demographics['age'])
                all_data.append({
                    'file': str(img_path.relative_to(test_dir)),
                    'age': age,
                    'gender': gender,
                    'race': race,
                    'service_test': 'True'
                })
                
                image_id += 1
    
    # Save metadata CSV
    df = pd.DataFrame(all_data)
    csv_path = test_dir / 'test_labels.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Created {len(all_data)} synthetic test images")
    print(f"✓ Saved metadata to: {csv_path}")
    print(f"✓ Test directory: {test_dir}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Gender distribution:")
    for gender in demographics['gender']:
        count = len(df[df['gender'] == gender])
        print(f"    {gender}: {count}")
    
    print(f"\n  Race distribution:")
    for race in demographics['race']:
        count = len(df[df['race'] == race])
        print(f"    {race}: {count}")
    
    return test_dir, csv_path

if __name__ == "__main__":
    test_dir, csv_path = create_test_dataset(num_samples_per_class=50)
    
    print("\n" + "="*60)
    print("SYNTHETIC TEST DATA READY")
    print("="*60)
    print("\nNow you can run:")
    print(f"  python fairness_eval.py --data_path {test_dir}")
    print(f"  python attack_sweep.py --data_path {test_dir}")