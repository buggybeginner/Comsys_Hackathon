import os
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import argparse

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Siamese Network ----------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ---------------- Image Utils ----------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_image_embedding(model, img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.forward_once(img).squeeze()

# ---------------- Reference Embedding ----------------
def build_reference_embeddings(model, train_dir, transform):
    ref_embeddings = {}
    for identity in os.listdir(train_dir):
        id_path = os.path.join(train_dir, identity)
        if not os.path.isdir(id_path):
            continue
        image_paths = glob(os.path.join(id_path, "*.jpg"))
        embeddings = [get_image_embedding(model, p, transform) for p in image_paths]
        if embeddings:
            ref_embeddings[identity] = torch.stack(embeddings)
    return ref_embeddings

# ---------------- Matching ----------------
def match_identity(embedding, ref_embeddings, threshold=1.0):
    min_dist = float('inf')
    matched_id = None
    for identity, embeds in ref_embeddings.items():
        dists = torch.norm(embedding - embeds, dim=1)
        d_min = dists.min().item()
        if d_min < min_dist:
            min_dist = d_min
            matched_id = identity
    if min_dist < threshold:
        return matched_id, 1, min_dist
    else:
        return None, 0, min_dist

# ---------------- Evaluation ----------------
def evaluate(model_path, train_dir, val_dir, threshold=1.0, output_csv="val_results.csv"):
    print(f"\n📦 Loading model from {model_path}")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transform()

    print(f"🔍 Building reference embeddings from {train_dir}")
    ref_embeddings = build_reference_embeddings(model, train_dir, transform)

    print(f"📂 Scanning distorted images in {val_dir}")
    distorted_images = []
    for identity in os.listdir(val_dir):
        distortion_folder = os.path.join(val_dir, identity, "distortion")
        if os.path.exists(distortion_folder):
            for img_path in glob(os.path.join(distortion_folder, "*.jpg")):
                distorted_images.append((img_path, identity))

    print(f"🔁 Evaluating {len(distorted_images)} images")
    y_true, y_pred, sim_scores, img_paths = [], [], [], []

    for img_path, true_id in tqdm(distorted_images):
        embedding = get_image_embedding(model, img_path, transform)
        predicted_id, pred_label, score = match_identity(embedding, ref_embeddings, threshold)

        y_true.append(1 if predicted_id == true_id else 0)
        y_pred.append(pred_label)
        sim_scores.append(score)
        img_paths.append(img_path)

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall   : {rec:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")

    results = pd.DataFrame({
        "image_path": img_paths,
        "ground_truth": y_true,
        "predicted": y_pred,
        "similarity_score": sim_scores
    })
    results.to_csv(output_csv, index=False)
    print(f"📁 Saved results to {output_csv}")

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--output_csv', type=str, default='val_results.csv')

    args = parser.parse_args()
    evaluate(args.model_path, args.train_dir, args.val_dir, args.threshold, args.output_csv)
