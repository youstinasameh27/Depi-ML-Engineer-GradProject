import kagglehub
import shutil
from pathlib import Path
import os

# Download dataset from KaggleHub
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
print("Dataset downloaded to:", path)

# Copy dataset from KaggleHub cache to your local project directory
target_dir = Path(r"C:\Users\Malak\MyFiles\Depi_project")
src_dir = Path(path)
deepfunneled = src_dir / "lfw-deepfunneled"

if deepfunneled.exists():
    shutil.copytree(deepfunneled, target_dir / "lfw-deepfunneled", dirs_exist_ok=True)
    print("Dataset copied to:", target_dir)
else:
    print("lfw-deepfunneled folder not found inside KaggleHub dataset.")

                                       ##### Dataset Preprocessing #####
import hashlib
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split

# Correct dataset root path based on local directory
base_dir = r'C:\Users\Malak\MyFiles\Depi_project\lfw-deepfunneled'
processed_dir = r'C:\Users\Malak\MyFiles\Depi_project\processed_faces'
train_dir = r'C:\Users\Malak\MyFiles\Depi_project\data_split\train'
val_dir = r'C:\Users\Malak\MyFiles\Depi_project\data_split\val'
val_split = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=10, min_face_size=10, device=device)

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

shutil.rmtree(processed_dir, ignore_errors=True)
os.makedirs(processed_dir, exist_ok=True)

# Test face detection on sample image to verify correct paths and model working
def test_face_detection(image_path):
    img = Image.open(image_path).convert('RGB')
    boxes, probs = mtcnn.detect(img)
    print(f"Detected boxes for {image_path}: {boxes}")

test_face_detection(r'C:\Users\Malak\MyFiles\Depi_project\lfw-deepfunneled\lfw-deepfunneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg')

# Step 1: Detect, crop, align faces, save
def save_faces(data_dir, save_dir):
    image_paths = list(Path(data_dir).rglob('*.jpg'))
    for img_path in tqdm(image_paths, desc="Processing images"):
        person_name = img_path.parent.name
        save_person_dir = Path(save_dir) / person_name
        save_person_dir.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                face_img = transforms.ToPILImage()(face)
                dst_path = save_person_dir / img_path.name
                face_img.save(dst_path)
                print(f"Saved face to {dst_path}")
            else:
                print(f"No face detected in {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

save_faces(base_dir, processed_dir)

num_processed = sum(1 for _ in Path(processed_dir).rglob('*.jpg'))
print(f"Number of processed face images: {num_processed}")

# Step 2: Augmentation
def augment_faces(src_dir, num_aug=2):
    for person_dir in Path(src_dir).iterdir():
        for img_path in person_dir.glob('*.jpg'):
            img = Image.open(img_path)
            for i in range(num_aug):
                img_aug = augment(img)
                save_aug_dir = Path(src_dir) / person_dir.name
                save_aug_dir.mkdir(parents=True, exist_ok=True)
                img_aug.save(save_aug_dir / f"{img_path.stem}_aug{i}{img_path.suffix}")

augment_faces(processed_dir)

# Step 3: Remove duplicates
def remove_duplicates(folder):
    seen_hashes = set()
    num_removed = 0
    for person_dir in Path(folder).iterdir():
        if not person_dir.is_dir():
            continue
        for img_path in list(person_dir.glob('*.jpg')):
            with open(img_path, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in seen_hashes:
                img_path.unlink()
                num_removed += 1
            else:
                seen_hashes.add(filehash)
    print(f"Removed {num_removed} duplicate images.")

remove_duplicates(processed_dir)

# Step 4: Train/val split
def split_train_val(src_dir, train_dir, val_dir, val_split=0.2):
    for person_dir in Path(src_dir).iterdir():
        img_paths = list(person_dir.glob('*.jpg'))
        if len(img_paths) < 2:
            continue
        train_imgs, val_imgs = train_test_split(img_paths, test_size=val_split, random_state=42)
        for t in train_imgs:
            out_dir = Path(train_dir) / person_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(t, out_dir / t.name)
        for v in val_imgs:
            out_dir = Path(val_dir) / person_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(v, out_dir / v.name)

split_train_val(processed_dir, train_dir, val_dir, val_split)

print("Clean and structured dataset ready for training.")
print(f"Train images in {train_dir}")
print(f"Validation images in {val_dir}")
                                      
                                       ##### Model Architecture & Traning #####

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset Setup
train_dir = r'C:\Users\Malak\MyFiles\Depi_project\data_split\train'
val_dir = r'C:\Users\Malak\MyFiles\Depi_project\data_split\val'

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Selection
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Model Training
num_epochs = 6
best_acc = 0

history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': []}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    preds, labels = [], []

    for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        if imgs.size(0) == 1:
            continue
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    train_acc = accuracy_score(labels, preds)
    history['train_loss'].append(train_loss/len(train_loader))
    history['train_acc'].append(train_acc*100)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}%")

    # Validation
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            if imgs.size(0) == 1:
                continue
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            val_labels.extend(targets.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    prec = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
    rec = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
    f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

    history['val_acc'].append(acc*100)
    history['val_prec'].append(prec*100)
    history['val_rec'].append(rec*100)
    history['val_f1'].append(f1*100)

    print(f"Validation | Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), r'C:\Users\Malak\MyFiles\Depi_project\best_facenet_model.pth')
        print("Model saved with better accuracy.")

# Model Evaluation and FAR Calculation
def false_acceptance_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    false_accepts = cm.sum(axis=0) - np.diag(cm)
    true_negatives = cm.sum() - cm.sum(axis=1) - cm.sum(axis=0) + np.diag(cm)
    far = false_accepts.sum() / (false_accepts.sum() + true_negatives.sum())
    return far

model.load_state_dict(torch.load(r'C:\Users\Malak\MyFiles\Depi_project\best_facenet_model.pth'))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, targets in val_loader:
        if imgs.size(0) == 1:
            continue
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
far = false_acceptance_rate(all_labels, all_preds)

print("\n===== Final Evaluation =====")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print(f"False Acceptance Rate (FAR): {far*100:.2f}%")

# 6. Visualization - Loss & Metrics
epochs = range(1, num_epochs+1)
plt.figure(figsize=(15,6))

# Loss
plt.subplot(1,2,1)
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Metrics
plt.subplot(1,2,2)
plt.plot(epochs, history['train_acc'], label='Train Acc')
plt.plot(epochs, history['val_acc'], label='Val Acc')
plt.plot(epochs, history['val_prec'], label='Precision')
plt.plot(epochs, history['val_rec'], label='Recall')
plt.plot(epochs, history['val_f1'], label='F1 Score')
plt.title('Performance Metrics per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Percentage (%)')
plt.legend()

plt.tight_layout()
plt.show()
