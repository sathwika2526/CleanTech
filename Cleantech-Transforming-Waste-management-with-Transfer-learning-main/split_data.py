import os
import shutil
import random

source_dir = source_dir = "C:/Users/ratna/OneDrive/Desktop/Waste Management/dataset"

dest_dir = dest_dir = r"C:\Users\ratna\OneDrive\Desktop\Waste Management\Waste_Classification_Dataset"

split_ratio = 0.8  # 80% for training

classes = ['biodegradable', 'recyclable', 'trash']

# Create destination folders
for split in ['train', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

# Split and copy files
for cls in classes:
    cls_dir = os.path.join(source_dir, cls)
    images = os.listdir(cls_dir)
    random.shuffle(images)
    
    split_point = int(len(images) * split_ratio)
    train_imgs = images[:split_point]
    test_imgs = images[split_point:]

    for img in train_imgs:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(dest_dir, 'train', cls, img)
        shutil.copyfile(src, dst)

    for img in test_imgs:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(dest_dir, 'test', cls, img)
        shutil.copyfile(src, dst)

print("✅ Dataset split into train and test folders successfully.")
