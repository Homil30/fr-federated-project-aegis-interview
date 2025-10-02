import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform else transforms.ToTensor()

        # Encode gender (Male=0, Female=1)
        self.annotations["gender_label"] = self.annotations["gender"].map({"Male": 0, "Female": 1})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, row["file"])
        image = Image.open(img_path).convert("RGB")
        label = row["gender_label"]

        if self.transform:
            image = self.transform(image)

        return image, label
