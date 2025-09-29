# inspect_data.py
import os
from torchvision import datasets

def inspect_folder(path):
    train = os.path.join(path, "train")
    val = os.path.join(path, "val")
    print("DATA ROOT:", path)
    if os.path.isdir(train):
        print("Found train/ folder.")
        ds = datasets.ImageFolder(train)
        print("Train classes (%d): %s" % (len(ds.classes), ds.classes))
    else:
        print("No train/ folder found at", train)
    if os.path.isdir(val):
        ds2 = datasets.ImageFolder(val)
        print("Val classes (%d): %s" % (len(ds2.classes), ds2.classes))
    else:
        print("No val/ folder found at", val)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=r"C:/Users/Asus/Desktop/XAI_Facial_Recognition/data/raw/FairFace")
    args = parser.parse_args()
    inspect_folder(args.data_dir)
